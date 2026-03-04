#!/usr/bin/env python3
"""
CUBE Batch Runner — GPU 1개당 N개 실험을 순서대로 실행.

사용법:
    CUDA_VISIBLE_DEVICES=1 python experiments/run_batch.py --gpu_id 1 --n_runs 10 --offset 0
    CUDA_VISIBLE_DEVICES=2 python experiments/run_batch.py --gpu_id 2 --n_runs 10 --offset 4
    CUDA_VISIBLE_DEVICES=3 python experiments/run_batch.py --gpu_id 3 --n_runs 10 --offset 8

각 GPU는 ALL_EXPERIMENTS 리스트에서 offset 위치부터 n_runs개 조합을 순차 실행.
같은 (baseline, budget)이 반복되면 타임스탬프 기반 run_id로 새 CSV 생성.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ALL_EXPERIMENTS = [
    ("reinforce", "none"),
    ("reinforce", "prompt_skip"),
    ("reinforce", "rollout_alloc"),
    ("reinforce", "subset_select"),
    ("grpo",      "none"),
    ("grpo",      "prompt_skip"),
    ("grpo",      "rollout_alloc"),
    ("grpo",      "subset_select"),
    ("rloo",      "none"),
    ("rloo",      "prompt_skip"),
    ("rloo",      "rollout_alloc"),
    ("rloo",      "subset_select"),
    ("stv",       "none"),
    ("stv",       "prompt_skip"),
    ("stv",       "rollout_alloc"),
    ("stv",       "subset_select"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id",          type=int, required=True)
    parser.add_argument("--n_runs",          type=int, default=15)
    parser.add_argument("--offset",          type=int, default=0,
                        help="ALL_EXPERIMENTS에서 시작할 인덱스 (mod 16)")
    parser.add_argument("--results_dir",     type=str, default="experiments/results")
    parser.add_argument("--M",               type=int, default=256)
    parser.add_argument("--B",               type=int, default=32)
    parser.add_argument("--N",               type=int, default=8)
    parser.add_argument("--S",               type=int, default=4)
    parser.add_argument("--K",               type=int, default=2)
    parser.add_argument("--T",               type=int, default=10)
    parser.add_argument("--R",               type=int, default=32)
    parser.add_argument("--num_train_steps", type=int, default=200)
    parser.add_argument("--lr",              type=float, default=1e-3)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    n = len(ALL_EXPERIMENTS)

    print(f"\nCUBE Batch Runner — GPU {args.gpu_id}  ({args.n_runs} runs, offset={args.offset})")
    print("=" * 60)

    for i in range(args.n_runs):
        baseline, budget = ALL_EXPERIMENTS[(args.offset + i) % n]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{ts}_{baseline}_{budget}_gpu{args.gpu_id}"

        print(f"\n[{i+1}/{args.n_runs}] GPU {args.gpu_id}: {baseline} × {budget}")
        print(f"  run_id: {run_id}")

        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_pilot.py"),
            "--baseline",        baseline,
            "--budget",          budget,
            "--gpu_id",          "0",          # CUDA_VISIBLE_DEVICES로 이미 지정
            "--run_id",          run_id,
            "--output_dir",      str(results_dir),
            "--M",               str(args.M),
            "--B",               str(args.B),
            "--N",               str(args.N),
            "--S",               str(args.S),
            "--K",               str(args.K),
            "--T",               str(args.T),
            "--R",               str(args.R),
            "--num_train_steps", str(args.num_train_steps),
            "--lr",              str(args.lr),
        ]

        import time
        t0 = time.time()
        ret = subprocess.run(cmd)
        elapsed = time.time() - t0
        status = "✓ DONE" if ret.returncode == 0 else "✗ FAILED"
        print(f"  {status} in {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print(f"GPU {args.gpu_id}: 완료 ({args.n_runs}/{args.n_runs})")


if __name__ == "__main__":
    main()
