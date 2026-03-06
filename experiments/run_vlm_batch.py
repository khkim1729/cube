#!/usr/bin/env python3
"""
CUBE VLM Batch Runner — distributes 16 (baseline × budget) combos across 3 GPUs.

Usage:
    # Launch all 3 GPUs in parallel (run each in a separate terminal):
    CUDA_VISIBLE_DEVICES=1 python experiments/run_vlm_batch.py --gpu_id 1 --offset 0
    CUDA_VISIBLE_DEVICES=2 python experiments/run_vlm_batch.py --gpu_id 2 --offset 5
    CUDA_VISIBLE_DEVICES=3 python experiments/run_vlm_batch.py --gpu_id 3 --offset 10

Each GPU runs its assigned combos sequentially.
Logs written to /tmp/cube_vlm_gpu{N}.log
"""

import argparse
import subprocess
import sys
import time
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
    parser = argparse.ArgumentParser(description="CUBE VLM multi-GPU batch runner")
    parser.add_argument("--gpu_id",          type=int, required=True,
                        help="Physical GPU index (used for CUDA_VISIBLE_DEVICES isolation)")
    parser.add_argument("--n_runs",          type=int, default=None,
                        help="Number of combos to run (default: remaining from offset)")
    parser.add_argument("--offset",          type=int, default=0,
                        help="Start index into ALL_EXPERIMENTS (0/5/10 for GPU 1/2/3)")
    parser.add_argument("--dataset",         type=str, default="mathvista",
                        choices=["mathvista", "mmstar", "chartqa", "scienceqa", "mmbench"])
    parser.add_argument("--results_dir",     type=str, default="experiments/results_vlm")
    parser.add_argument("--lora_rank",       type=int, default=16)
    parser.add_argument("--M",               type=int, default=256)
    parser.add_argument("--B",               type=int, default=32)
    parser.add_argument("--N",               type=int, default=8)
    parser.add_argument("--S",               type=int, default=4)
    parser.add_argument("--K",               type=int, default=2)
    parser.add_argument("--T",               type=int, default=10)
    parser.add_argument("--R",               type=int, default=32)
    parser.add_argument("--num_train_steps", type=int, default=200)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--n_pool",          type=int, default=512)
    args = parser.parse_args()

    n_total = len(ALL_EXPERIMENTS)
    # Default: take 5-6 combos per GPU (ceil(16/3))
    if args.n_runs is None:
        args.n_runs = min(6, n_total - args.offset)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(f"/tmp/cube_vlm_gpu{args.gpu_id}.log")
    print(f"\nCUBE VLM Batch Runner — GPU {args.gpu_id}")
    print(f"  combos: {args.n_runs} starting at offset {args.offset}")
    print(f"  dataset: {args.dataset}")
    print(f"  log: {log_path}")
    print("=" * 60)

    run_script = str(Path(__file__).parent / "run_vlm.py")

    for i in range(args.n_runs):
        idx = (args.offset + i) % n_total
        baseline, budget = ALL_EXPERIMENTS[idx]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{ts}_{baseline}_{budget}_{args.dataset}_gpu{args.gpu_id}"

        print(f"\n[{i+1}/{args.n_runs}] GPU {args.gpu_id}: {baseline} × {budget}")
        print(f"  run_id: {run_id}")

        cmd = [
            sys.executable, run_script,
            "--baseline",        baseline,
            "--budget",          budget,
            "--dataset",         args.dataset,
            "--gpu_id",          "0",    # CUDA_VISIBLE_DEVICES already set
            "--run_id",          run_id,
            "--output_dir",      str(results_dir),
            "--lora_rank",       str(args.lora_rank),
            "--M",               str(args.M),
            "--B",               str(args.B),
            "--N",               str(args.N),
            "--S",               str(args.S),
            "--K",               str(args.K),
            "--T",               str(args.T),
            "--R",               str(args.R),
            "--num_train_steps", str(args.num_train_steps),
            "--lr",              str(args.lr),
            "--n_pool",          str(args.n_pool),
        ]

        t0 = time.time()
        with open(log_path, "a") as log_f:
            log_f.write(f"\n{'='*60}\n")
            log_f.write(f"[{datetime.now().isoformat()}] {baseline} × {budget}\n")
            log_f.flush()
            ret = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0

        status = "DONE" if ret.returncode == 0 else f"FAILED (rc={ret.returncode})"
        print(f"  {status} in {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print(f"GPU {args.gpu_id}: completed ({args.n_runs}/{args.n_runs})")


if __name__ == "__main__":
    main()
