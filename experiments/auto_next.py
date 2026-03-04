#!/usr/bin/env python3
"""
CUBE Auto-Scheduler — 안 돌아간 실험을 자동으로 감지하고 실행.

사용법 (터미널 3개):
    CUDA_VISIBLE_DEVICES=1 python experiments/auto_next.py --gpu_id 1
    CUDA_VISIBLE_DEVICES=2 python experiments/auto_next.py --gpu_id 2
    CUDA_VISIBLE_DEVICES=3 python experiments/auto_next.py --gpu_id 3

동작 방식:
  1. experiments/results/ 의 CSV 파일을 스캔하여 완료된 실험 파악
  2. queue.json에서 현재 다른 GPU가 실행 중인 실험 제외
  3. 완료 안된 실험 중 가장 위에 있는 것을 선택
  4. PID 기록 → 실험 실행 → 완료 표시
  5. 완료까지 자동 반복 (큐 전부 소진 시 종료)

중단 복구:
  - "running" 상태인데 PID가 죽어있으면 → pending으로 자동 복원
  - CSV rows < T (=10) 이면 partial로 간주, 재실행

동시성:
  - fcntl.LOCK_EX로 queue.json 독점 잠금
  - 다른 터미널이 같은 실험을 중복 선택하는 것 방지
"""

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 전체 12개 실험 큐 (실행 우선순위 순)
# ─────────────────────────────────────────────────────────────────────────────
ALL_EXPERIMENTS = [
    {"baseline": "reinforce", "budget": "none"},
    {"baseline": "reinforce", "budget": "prompt_skip"},
    {"baseline": "reinforce", "budget": "rollout_alloc"},
    {"baseline": "reinforce", "budget": "subset_select"},
    {"baseline": "grpo",      "budget": "none"},
    {"baseline": "grpo",      "budget": "prompt_skip"},
    {"baseline": "grpo",      "budget": "rollout_alloc"},
    {"baseline": "grpo",      "budget": "subset_select"},
    {"baseline": "rloo",      "budget": "none"},
    {"baseline": "rloo",      "budget": "prompt_skip"},
    {"baseline": "rloo",      "budget": "rollout_alloc"},
    {"baseline": "rloo",      "budget": "subset_select"},
    {"baseline": "stv",       "budget": "none"},
    {"baseline": "stv",       "budget": "prompt_skip"},
    {"baseline": "stv",       "budget": "rollout_alloc"},
    {"baseline": "stv",       "budget": "subset_select"},
]

# 상태 상수
STATUS_PENDING   = "pending"
STATUS_RUNNING   = "running"
STATUS_COMPLETED = "completed"


# ─────────────────────────────────────────────────────────────────────────────
# 큐 파일 경로
# ─────────────────────────────────────────────────────────────────────────────
def queue_path(results_dir: Path) -> Path:
    return results_dir / "queue.json"

def lock_path(results_dir: Path) -> Path:
    return results_dir / "queue.lock"


# ─────────────────────────────────────────────────────────────────────────────
# CSV 스캔: 완료된 (baseline, budget) 파악
# ─────────────────────────────────────────────────────────────────────────────
def scan_results(results_dir: Path, T: int = 10) -> dict:
    """CSV 파일을 스캔하여 각 (baseline, budget) 조합의 완료된 체크포인트 수 반환.

    Returns:
        dict: {(baseline, budget): max_rows}
    """
    counts = {}
    for csv_file in sorted(results_dir.glob("*.csv")):
        if csv_file.name in ("combined_results.csv",):
            continue
        try:
            with open(csv_file, newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            bl = rows[0].get("baseline", "")
            bg = rows[0].get("budget", "")
            if bl and bg:
                key = (bl, bg)
                counts[key] = max(counts.get(key, 0), len(rows))
        except Exception:
            continue
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# 큐 초기화 / 로드 / 상태 업데이트
# ─────────────────────────────────────────────────────────────────────────────
def load_or_init_queue(results_dir: Path, T: int = 10) -> list:
    """큐를 로드하거나 없으면 생성. CSV 스캔 결과로 상태 갱신."""
    qp = queue_path(results_dir)
    completion = scan_results(results_dir, T)

    if qp.exists():
        with open(qp) as f:
            queue = json.load(f)
    else:
        queue = [
            {
                "id": f"{e['baseline']}_{e['budget']}",
                "baseline": e["baseline"],
                "budget": e["budget"],
                "status": STATUS_PENDING,
                "gpu_id": None,
                "pid": None,
                "run_id": None,
                "started_at": None,
                "completed_at": None,
                "csv_rows": 0,
            }
            for e in ALL_EXPERIMENTS
        ]

    # CSV 기반으로 상태 갱신
    for entry in queue:
        key = (entry["baseline"], entry["budget"])
        rows = completion.get(key, 0)
        entry["csv_rows"] = rows

        if rows >= T:
            # 완료된 실험
            entry["status"] = STATUS_COMPLETED
        elif entry["status"] == STATUS_RUNNING:
            # "running"인데 PID가 죽어있으면 pending으로 복원
            pid = entry.get("pid")
            if pid and not is_process_alive(pid):
                print(f"  [queue] '{entry['id']}' was running (PID={pid}) but process is dead → reset to pending")
                entry["status"] = STATUS_PENDING
                entry["gpu_id"] = None
                entry["pid"] = None
        elif rows > 0 and entry["status"] == STATUS_PENDING:
            # partial 실행 (중단된 경우) → 재실행
            print(f"  [queue] '{entry['id']}' has {rows}/{T} rows (partial) → will re-run")

    return queue


def save_queue(results_dir: Path, queue: list):
    """큐를 JSON으로 저장."""
    with open(queue_path(results_dir), "w") as f:
        json.dump(queue, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# 프로세스 생존 확인
# ─────────────────────────────────────────────────────────────────────────────
def is_process_alive(pid: int) -> bool:
    """PID가 살아있으면 True."""
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True  # 존재하지만 권한 없음 → 살아있음


# ─────────────────────────────────────────────────────────────────────────────
# 큐에서 다음 실험 선택 (잠금 보호)
# ─────────────────────────────────────────────────────────────────────────────
def claim_next_experiment(results_dir: Path, gpu_id: int, T: int = 10):
    """파일 잠금으로 보호된 상태에서 다음 pending 실험을 claim.

    Returns:
        dict or None: claimed experiment entry, or None if nothing pending
    """
    lp = lock_path(results_dir)
    lp.touch(exist_ok=True)

    with open(lp, "r+") as lock_file:
        # 독점 잠금 (다른 프로세스가 동시에 claim하는 것 방지)
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            queue = load_or_init_queue(results_dir, T)

            # 다음 pending 실험 선택
            claimed = None
            for entry in queue:
                if entry["status"] == STATUS_PENDING:
                    claimed = entry
                    break

            if claimed is None:
                save_queue(results_dir, queue)
                return None

            # 선택한 실험을 running으로 마킹
            claimed["status"]     = STATUS_RUNNING
            claimed["gpu_id"]     = gpu_id
            claimed["pid"]        = os.getpid()
            claimed["started_at"] = datetime.now().isoformat()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            claimed["run_id"]     = f"{ts}_{claimed['baseline']}_{claimed['budget']}_gpu{gpu_id}"

            save_queue(results_dir, queue)
            return dict(claimed)  # 복사본 반환

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def mark_experiment(results_dir: Path, exp_id: str, status: str):
    """실험 상태를 completed / pending으로 업데이트 (잠금 보호)."""
    lp = lock_path(results_dir)
    lp.touch(exist_ok=True)

    with open(lp, "r+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            queue = load_or_init_queue(results_dir)
            for entry in queue:
                if entry["id"] == exp_id:
                    entry["status"] = status
                    if status == STATUS_COMPLETED:
                        entry["completed_at"] = datetime.now().isoformat()
                    elif status == STATUS_PENDING:
                        entry["gpu_id"] = None
                        entry["pid"]    = None
                    break
            save_queue(results_dir, queue)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


# ─────────────────────────────────────────────────────────────────────────────
# 단일 실험 실행
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(entry: dict, gpu_id: int, results_dir: str, args) -> bool:
    """run_pilot.py를 subprocess로 실행. 성공 시 True 반환."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_pilot.py"),
        "--baseline",    entry["baseline"],
        "--budget",      entry["budget"],
        "--gpu_id",      "0",         # CUDA_VISIBLE_DEVICES로 이미 GPU 지정됨
        "--output_dir",  results_dir,
        "--run_id",      entry["run_id"],
        "--M",           str(args.M),
        "--B",           str(args.B),
        "--N",           str(args.N),
        "--S",           str(args.S),
        "--K",           str(args.K),
        "--T",           str(args.T),
        "--R",           str(args.R),
        "--num_train_steps", str(args.num_train_steps),
        "--lr",          str(args.lr),
    ]

    print(f"\n{'='*60}")
    print(f"  GPU {gpu_id}: {entry['baseline']} × {entry['budget']}")
    print(f"  run_id: {entry['run_id']}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    start = time.time()
    proc = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    success = proc.returncode == 0
    status = "✓ DONE" if success else "✗ FAILED"
    print(f"  {status} in {elapsed:.0f}s — {entry['baseline']} × {entry['budget']}")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# 큐 상태 출력
# ─────────────────────────────────────────────────────────────────────────────
def print_queue_status(results_dir: Path, T: int = 10):
    """현재 큐 상태를 테이블로 출력."""
    queue = load_or_init_queue(results_dir, T)
    print(f"\n{'─'*65}")
    print(f"  {'ID':<30} {'Status':<12} {'GPU':<5} {'Rows':<8} {'PID'}")
    print(f"{'─'*65}")
    for e in queue:
        alive = ""
        if e["status"] == STATUS_RUNNING and e.get("pid"):
            alive = "(alive)" if is_process_alive(e["pid"]) else "(dead!)"
        pid_str = f"{e.get('pid','')} {alive}" if e.get("pid") else ""
        rows_str = f"{e['csv_rows']}/{T}"
        print(f"  {e['id']:<30} {e['status']:<12} {str(e.get('gpu_id','')):<5} {rows_str:<8} {pid_str}")
    n_done = sum(1 for e in queue if e["status"] == STATUS_COMPLETED)
    n_run  = sum(1 for e in queue if e["status"] == STATUS_RUNNING)
    n_pend = sum(1 for e in queue if e["status"] == STATUS_PENDING)
    print(f"{'─'*65}")
    print(f"  completed={n_done}  running={n_run}  pending={n_pend}  total={len(queue)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CUBE auto-scheduler")
    parser.add_argument("--gpu_id",       type=int, required=True,
                        help="이 워커가 사용할 GPU ID (1, 2, 3)")
    parser.add_argument("--results_dir",  type=str,
                        default="experiments/results")
    parser.add_argument("--status",       action="store_true",
                        help="큐 상태만 출력하고 종료")
    parser.add_argument("--reset",        action="store_true",
                        help="queue.json 삭제 후 새로 시작")

    # run_pilot.py에 전달할 실험 파라미터
    parser.add_argument("--M",               type=int,   default=256)
    parser.add_argument("--B",               type=int,   default=32)
    parser.add_argument("--N",               type=int,   default=8)
    parser.add_argument("--S",               type=int,   default=4,
                        help="Monte-Carlo 샘플 수 (full=16, pilot=4)")
    parser.add_argument("--K",               type=int,   default=2,
                        help="롤아웃 리샘플 수 (full=8, pilot=2)")
    parser.add_argument("--T",               type=int,   default=10)
    parser.add_argument("--R",               type=int,   default=32)
    parser.add_argument("--num_train_steps", type=int,   default=200)
    parser.add_argument("--lr",              type=float, default=1e-3)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 큐 초기화 리셋 ────────────────────────────────────────────────────
    if args.reset:
        qp = queue_path(results_dir)
        if qp.exists():
            qp.unlink()
            print(f"  Deleted {qp}")

    # ── 상태만 출력 ───────────────────────────────────────────────────────
    if args.status:
        print_queue_status(results_dir, args.T)
        return

    # ── 메인 스케줄링 루프 ────────────────────────────────────────────────
    print(f"\nCUBE Auto-Scheduler  (GPU {args.gpu_id}  PID={os.getpid()})")
    print_queue_status(results_dir, args.T)

    n_done_this_session = 0

    while True:
        entry = claim_next_experiment(results_dir, args.gpu_id, args.T)

        if entry is None:
            if n_done_this_session == 0:
                print(f"  GPU {args.gpu_id}: 남은 pending 실험 없음. 모든 실험이 완료되었거나 다른 GPU가 담당 중.")
            else:
                print(f"\n  GPU {args.gpu_id}: 모든 pending 실험 완료. (이번 세션: {n_done_this_session}개)")
            break

        exp_id = entry["id"]
        print(f"\n  [GPU {args.gpu_id}] 선택: {entry['baseline']} × {entry['budget']}  (queue id: {exp_id})")

        success = run_experiment(entry, args.gpu_id, str(results_dir), args)

        if success:
            mark_experiment(results_dir, exp_id, STATUS_COMPLETED)
            n_done_this_session += 1
        else:
            # 실패 시 pending으로 돌려서 다음 기회에 재시도
            print(f"  [GPU {args.gpu_id}] '{exp_id}' 실패 → pending으로 복원")
            mark_experiment(results_dir, exp_id, STATUS_PENDING)
            # 짧은 대기 후 재시도 방지를 위해 다음 실험으로 넘어감
            time.sleep(2)

    print_queue_status(results_dir, args.T)


if __name__ == "__main__":
    main()
