"""
CUBE 실험 자동 론처 — 3개 파일럿 실험을 GPU 1,2,3에서 병렬 실행.
파일럿 완료 후 나머지 9개 실험을 자동으로 큐잉하여 순차 실행.

실험 설계 (3 파일럿):
  GPU 1: RLOO × None           → 이론적으로 가장 깨끗한 조합 (편향 최소)
  GPU 2: GRPO × SubsetSelect   → 융합 편향 + 분산 증폭 예상
  GPU 3: STV × PromptSkip      → 크로스-프롬프트 잡음 라우팅 + 프롬프트 스킵 편향

자동 체이닝:
  파일럿 3개 완료 → 나머지 9개 (GPU 1,2,3에 3개씩) 자동 큐잉 실행

사용법:
    python experiments/launch_pilots.py [--output_dir experiments/results] [--dry_run]
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 전체 실험 큐 (4 baseline × 3 budget = 12 combinations)
# ─────────────────────────────────────────────────────────────────────────────

# 3 파일럿 (병렬, GPU 1,2,3)
PILOT_EXPERIMENTS = [
    {"baseline": "rloo",      "budget": "none",          "gpu": 1, "priority": "pilot"},
    {"baseline": "grpo",      "budget": "subset_select", "gpu": 2, "priority": "pilot"},
    {"baseline": "stv",       "budget": "prompt_skip",   "gpu": 3, "priority": "pilot"},
]

# 나머지 9개 (파일럿 완료 후 자동 실행, GPU 1,2,3에 3개씩)
REMAINING_EXPERIMENTS = [
    # 배치 1 (파일럿 완료 직후)
    {"baseline": "reinforce", "budget": "none",          "gpu": 1, "batch": 1},
    {"baseline": "grpo",      "budget": "none",          "gpu": 2, "batch": 1},
    {"baseline": "rloo",      "budget": "prompt_skip",   "gpu": 3, "batch": 1},
    # 배치 2
    {"baseline": "reinforce", "budget": "prompt_skip",   "gpu": 1, "batch": 2},
    {"baseline": "grpo",      "budget": "prompt_skip",   "gpu": 2, "batch": 2},
    {"baseline": "stv",       "budget": "none",          "gpu": 3, "batch": 2},
    # 배치 3
    {"baseline": "reinforce", "budget": "subset_select", "gpu": 1, "batch": 3},
    {"baseline": "rloo",      "budget": "subset_select", "gpu": 2, "batch": 3},
    {"baseline": "stv",       "budget": "subset_select", "gpu": 3, "batch": 3},
]

ALL_EXPERIMENTS = PILOT_EXPERIMENTS + REMAINING_EXPERIMENTS


# ─────────────────────────────────────────────────────────────────────────────
# 실험 실행 함수
# ─────────────────────────────────────────────────────────────────────────────

def run_single_experiment(
    baseline: str,
    budget: str,
    gpu_id: int,
    output_dir: str,
    run_id: str,
    extra_args: list = None,
    dry_run: bool = False,
) -> tuple:
    """단일 실험을 subprocess로 실행. (baseline, budget, success, elapsed)를 반환."""

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_pilot.py"),
        "--baseline", baseline,
        "--budget", budget,
        "--gpu_id", str(gpu_id),
        "--output_dir", output_dir,
        "--run_id", run_id,
    ]
    if extra_args:
        cmd.extend(extra_args)

    start = time.time()
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        time.sleep(0.5)
        return baseline, budget, True, 0.5

    print(f"  [LAUNCH] GPU:{gpu_id} {baseline} × {budget}  run_id={run_id}")
    try:
        proc = subprocess.run(cmd, capture_output=False, text=True, timeout=3600)
        success = proc.returncode == 0
        elapsed = time.time() - start
        if not success:
            print(f"  [FAIL]   GPU:{gpu_id} {baseline} × {budget}  (exit={proc.returncode})")
        else:
            print(f"  [DONE]   GPU:{gpu_id} {baseline} × {budget}  ({elapsed:.0f}s)")
        return baseline, budget, success, elapsed
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {baseline} × {budget}")
        return baseline, budget, False, 3600


def make_run_id(baseline: str, budget: str, gpu_id: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{baseline}_{budget}_gpu{gpu_id}"


# ─────────────────────────────────────────────────────────────────────────────
# 병렬 배치 실행
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_parallel(experiments: list, output_dir: str, dry_run: bool, extra_args: list = None) -> list:
    """여러 실험을 ProcessPoolExecutor로 병렬 실행."""
    results = []
    with ProcessPoolExecutor(max_workers=len(experiments)) as executor:
        futures = {}
        for exp in experiments:
            run_id = make_run_id(exp["baseline"], exp["budget"], exp["gpu"])
            future = executor.submit(
                run_single_experiment,
                exp["baseline"],
                exp["budget"],
                exp["gpu"],
                output_dir,
                run_id,
                extra_args,
                dry_run,
            )
            futures[future] = exp

        for future in as_completed(futures):
            exp = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"  [ERROR] {exp['baseline']} × {exp['budget']}: {e}")
                results.append((exp["baseline"], exp["budget"], False, 0))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV 병합
# ─────────────────────────────────────────────────────────────────────────────

def merge_csv(output_dir: str) -> Path:
    """모든 개별 CSV를 하나의 combined_results.csv로 병합."""
    import csv
    import glob

    out_dir = Path(output_dir)
    csv_files = sorted(glob.glob(str(out_dir / "*.csv")))
    # combined 파일 자체는 제외
    csv_files = [f for f in csv_files if "combined" not in f]

    if not csv_files:
        print("  No CSV files to merge.")
        return None

    combined_path = out_dir / "combined_results.csv"
    all_rows = []
    header = None

    for f in csv_files:
        with open(f, newline="") as fp:
            reader = csv.DictReader(fp)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    with open(combined_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  Merged {len(all_rows)} rows from {len(csv_files)} files → {combined_path}")
    return combined_path


# ─────────────────────────────────────────────────────────────────────────────
# Main Launcher
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CUBE multi-GPU experiment launcher")
    parser.add_argument("--output_dir",       type=str,   default="experiments/results")
    parser.add_argument("--dry_run",          action="store_true")
    parser.add_argument("--pilots_only",      action="store_true",
                        help="Run only 3 pilot experiments, skip auto-chaining")
    # Pass-through args to run_pilot.py
    parser.add_argument("--S",                type=int,   default=16)
    parser.add_argument("--K",                type=int,   default=8)
    parser.add_argument("--T",                type=int,   default=10)
    parser.add_argument("--R",                type=int,   default=32)
    parser.add_argument("--M",                type=int,   default=256)
    parser.add_argument("--B",                type=int,   default=32)
    parser.add_argument("--N",                type=int,   default=8)
    parser.add_argument("--num_train_steps",  type=int,   default=200)
    parser.add_argument("--lr",               type=float, default=1e-3)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    extra_args = [
        "--S",               str(args.S),
        "--K",               str(args.K),
        "--T",               str(args.T),
        "--R",               str(args.R),
        "--M",               str(args.M),
        "--B",               str(args.B),
        "--N",               str(args.N),
        "--num_train_steps", str(args.num_train_steps),
        "--lr",              str(args.lr),
    ]

    total_start = time.time()

    # ── 스텝 1: 3개 파일럿 실험 병렬 실행 ──────────────────────────────────
    print("=" * 70)
    print("CUBE Experiment Launcher")
    print(f"  Output: {args.output_dir}")
    print(f"  S={args.S}, K={args.K}, T={args.T}, R={args.R}, M={args.M}, B={args.B}")
    print("=" * 70)
    print("\n[Phase 1] Pilot experiments (GPU 1,2,3 — parallel):")
    print("  GPU 1: RLOO × None")
    print("  GPU 2: GRPO × SubsetSelect")
    print("  GPU 3: STV × PromptSkip")

    pilot_results = run_batch_parallel(PILOT_EXPERIMENTS, args.output_dir, args.dry_run, extra_args)
    n_success = sum(1 for _, _, ok, _ in pilot_results if ok)
    print(f"\n  Pilots done: {n_success}/3 succeeded")

    if args.pilots_only:
        merge_csv(args.output_dir)
        print(f"\nTotal elapsed: {time.time() - total_start:.0f}s")
        return

    # ── 스텝 2–4: 나머지 9개 실험 자동 체이닝 (배치 3개씩) ─────────────────
    for batch_num in [1, 2, 3]:
        batch_exps = [e for e in REMAINING_EXPERIMENTS if e["batch"] == batch_num]
        baselines_budgets = " | ".join(f"GPU{e['gpu']}: {e['baseline']}×{e['budget']}" for e in batch_exps)
        print(f"\n[Phase {batch_num + 1}] Remaining batch {batch_num} — {baselines_budgets}")
        batch_results = run_batch_parallel(batch_exps, args.output_dir, args.dry_run, extra_args)
        n_ok = sum(1 for _, _, ok, _ in batch_results if ok)
        print(f"  Batch {batch_num} done: {n_ok}/3 succeeded")

    # ── 최종: 모든 CSV 병합 ────────────────────────────────────────────────
    print("\n[Final] Merging all results...")
    combined = merge_csv(args.output_dir)

    total_elapsed = time.time() - total_start
    print(f"\nAll 12 experiments completed in {total_elapsed:.0f}s")
    if combined:
        print(f"Master CSV: {combined}")


if __name__ == "__main__":
    main()
