"""
CUBE Pilot Experiment Runner — 단일 (baseline × budget) 실험 실행.

사용법:
    python experiments/run_pilot.py \\
        --baseline rloo \\
        --budget none \\
        --gpu 1 \\
        --output_dir experiments/results \\
        --run_id pilot_rloo_none

plans_cube_02.txt 파라미터:
    M=256, B=32, N=8, S=16, K=8, T=10, R=32

결과는 CSV로 저장: <output_dir>/<run_id>.csv
각 행 = 하나의 체크포인트에서의 측정값
"""

import argparse
import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.cube_sim import (
    ToyPolicy, DataPool,
    measure_checkpoint, aggregate_metrics,
    build_A_B, build_H, sample_rollouts, build_D_B,
)
from cube.utils.probe import make_probe_vectors


# ─────────────────────────────────────────────────────────────────────────────
# CSV Schema (컬럼 이름 정의)
# ─────────────────────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    # ── 메타 정보 ──────────────────────────────────────────────────────────
    "run_id",
    "date",           # YYYY-MM-DD
    "time",           # HH:MM:SS
    "timestamp",      # ISO 8601
    "gpu_id",
    "baseline",       # reinforce / grpo / rloo / stv
    "budget",         # none / prompt_skip / rollout_alloc / subset_select
    # ── 하이퍼파라미터 ──────────────────────────────────────────────────────
    "step",           # 학습 스텝 (0 ~ num_train_steps)
    "checkpoint_idx", # 로깅 체크포인트 인덱스 (0 ~ T-1)
    "M",              # 총 롤아웃 수
    "B",              # 미니배치 프롬프트 수
    "N_per_prompt",   # 프롬프트당 롤아웃 수
    "S",              # 미니배치 샘플링 횟수
    "K",              # 롤아웃 리샘플 횟수
    "T",              # 총 로깅 체크포인트 수
    "R",              # probe 벡터 수
    "probe_seed",     # probe 벡터 생성 시드
    # ── 편향 분해 (Theorem 1) ──────────────────────────────────────────────
    "total_bias_norm",
    "total_bias_proj_mean",
    "total_bias_proj_std",
    "budget_bias_proj_mean",
    "budget_bias_proj_std",
    "baseline_bias_proj_mean",
    "baseline_bias_proj_std",
    "fusion_bias_proj_mean",
    "fusion_bias_proj_std",
    # ── 분산 분해 (Theorem 2) ──────────────────────────────────────────────
    "total_var_mean",
    "total_var_std",
    "within_var_mean",
    "within_var_std",
    "across_var_mean",
    "across_var_std",
    # ── HL Proxy (||HL||_F^2) ──────────────────────────────────────────────
    "HL_proxy_mean",
    "HL_proxy_std",
    # ── 그래디언트 / 보상 진단 ──────────────────────────────────────────────
    "g_hat_norm_mean",
    "g_hat_norm_std",
    "g_ref_norm_mean",
    "g_ref_norm_std",
    "reward_mean",
    "reward_std_mean",
    # ── 런타임 ─────────────────────────────────────────────────────────────
    "elapsed_seconds",
    "checkpoint_elapsed_seconds",
]


def write_csv_header(csv_path: Path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_csv_row(csv_path: Path, row: dict):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Training Step (실제 파라미터 업데이트)
# ─────────────────────────────────────────────────────────────────────────────

def train_step(
    model: ToyPolicy,
    optimizer: optim.Optimizer,
    data_pool: DataPool,
    baseline: str,
    budget: str,
    B: int,
    N: int,
    rng: torch.Generator,
    device: str,
):
    """One policy gradient update step using the specified baseline+budget stack."""
    prompts_b, golds_b = data_pool.sample_minibatch(B, rng)
    rollouts = sample_rollouts(model, prompts_b, golds_b, N, rng, device)
    r = rollouts.rewards  # (M,)
    M = rollouts.M

    # Build operators
    A_B = build_A_B(rollouts, baseline, r)
    d_B = build_D_B(rollouts, baseline, r)
    H, H0 = build_H(rollouts, budget, r)
    A_B_r = A_B @ r
    tilde_a = d_B * (r - A_B_r)  # (M,) advantage

    # Compute log_probs for all rollouts (with grad)
    logits = model(rollouts.prompts)  # (M, n_classes)
    log_probs = torch.log_softmax(logits, dim=-1)
    lp_selected = log_probs[range(M), rollouts.answers]  # (M,)

    # Policy gradient loss: -sum h_m * tilde_a_m * log_pi_m
    loss = -(lp_selected * (H * tilde_a)).sum()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), r.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)

    print(f"[{args.run_id}] GPU:{args.gpu_id}  {args.baseline} × {args.budget}")
    print(f"  Device: {torch.cuda.get_device_name(args.gpu_id)}")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.run_id}.csv"
    meta_path = out_dir / f"{args.run_id}_meta.json"

    # ── Probe vectors (fixed for all experiments) ─────────────────────────────
    # Build model first to get d
    model = ToyPolicy(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        n_classes=args.n_classes,
        seed=42,
    ).to(device)
    d = model.n_params
    print(f"  Model params: d = {d}")

    probe_vecs = make_probe_vectors(d, R=args.R, seed=args.probe_seed, device=device)
    print(f"  Probe vectors: (R={args.R}, d={d}), seed={args.probe_seed}")

    # ── Data pool ─────────────────────────────────────────────────────────────
    data_pool = DataPool(
        n_pool=args.n_pool,
        input_dim=args.input_dim,
        n_classes=args.n_classes,
        seed=args.data_seed,
        device=device,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.train_seed)

    # ── CSV & Meta ─────────────────────────────────────────────────────────────
    write_csv_header(csv_path)
    meta = {
        "run_id": args.run_id,
        "baseline": args.baseline,
        "budget": args.budget,
        "gpu_id": args.gpu_id,
        "d": d,
        "M": args.M, "B": args.B, "N": args.N,
        "S": args.S, "K": args.K, "T": args.T, "R": args.R,
        "probe_seed": args.probe_seed,
        "num_train_steps": args.num_train_steps,
        "lr": args.lr,
        "start_time": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── Training & Logging Loop ───────────────────────────────────────────────
    log_interval = args.num_train_steps // args.T  # every N steps → 1 checkpoint
    total_start = time.time()
    checkpoint_idx = 0

    for step in range(args.num_train_steps + 1):
        # ── Log at checkpoint ──────────────────────────────────────────────
        if step % log_interval == 0 and checkpoint_idx < args.T:
            ckpt_start = time.time()
            dt = datetime.now()

            print(f"  [step={step:4d}] Measuring checkpoint {checkpoint_idx}/{args.T-1}...")

            # Freeze model, measure bias/variance (plans_cube_02.txt 프로토콜)
            model.eval()
            with torch.enable_grad():  # need grad for psi computation
                cm = measure_checkpoint(
                    model=model,
                    data_pool=data_pool,
                    baseline=args.baseline,
                    budget=args.budget,
                    probe_vecs=probe_vecs,
                    S=args.S,
                    K=args.K,
                    B=args.B,
                    N=args.N,
                    probe_seed_base=args.probe_seed + step,
                    device=device,
                )
            model.train()

            metrics = aggregate_metrics(cm)
            ckpt_elapsed = time.time() - ckpt_start
            total_elapsed = time.time() - total_start

            row = {
                "run_id": args.run_id,
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M:%S"),
                "timestamp": dt.isoformat(),
                "gpu_id": args.gpu_id,
                "baseline": args.baseline,
                "budget": args.budget,
                "step": step,
                "checkpoint_idx": checkpoint_idx,
                "M": args.M, "B": args.B, "N_per_prompt": args.N,
                "S": args.S, "K": args.K, "T": args.T, "R": args.R,
                "probe_seed": args.probe_seed,
                "elapsed_seconds": round(total_elapsed, 2),
                "checkpoint_elapsed_seconds": round(ckpt_elapsed, 2),
                **{k: round(v, 6) for k, v in metrics.items()},
            }
            append_csv_row(csv_path, row)

            print(
                f"    total_bias={metrics['total_bias_norm']:.4f} | "
                f"fusion_bias={metrics['fusion_bias_proj_mean']:.4f} | "
                f"total_var={metrics['total_var_mean']:.4f} | "
                f"HL={metrics['HL_proxy_mean']:.4f} | "
                f"reward={metrics['reward_mean']:.3f} | "
                f"t={ckpt_elapsed:.1f}s"
            )
            checkpoint_idx += 1

        # ── Train step ────────────────────────────────────────────────────
        if step < args.num_train_steps:
            loss, rew = train_step(
                model, optimizer, data_pool,
                args.baseline, args.budget,
                args.B, args.N, rng, device,
            )

    total_elapsed = time.time() - total_start
    print(f"  [{args.run_id}] Done in {total_elapsed:.1f}s → {csv_path}")

    # Update meta with end time
    meta["end_time"] = datetime.now().isoformat()
    meta["total_elapsed_seconds"] = round(total_elapsed, 2)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CUBE pilot experiment runner")
    # Experiment identity
    p.add_argument("--baseline",    type=str, required=True,
                   choices=["reinforce", "grpo", "rloo", "stv"])
    p.add_argument("--budget",      type=str, required=True,
                   choices=["none", "prompt_skip", "rollout_alloc", "subset_select"])
    p.add_argument("--gpu_id",      type=int, default=1)
    p.add_argument("--run_id",      type=str, default=None)
    p.add_argument("--output_dir",  type=str, default="experiments/results")

    # plans_cube_02.txt parameters
    p.add_argument("--M",           type=int, default=256)
    p.add_argument("--B",           type=int, default=32)
    p.add_argument("--N",           type=int, default=8)    # M/B
    p.add_argument("--S",           type=int, default=16)
    p.add_argument("--K",           type=int, default=8)
    p.add_argument("--T",           type=int, default=10)
    p.add_argument("--R",           type=int, default=32)
    p.add_argument("--probe_seed",  type=int, default=42)

    # Model / training
    p.add_argument("--input_dim",   type=int, default=64)
    p.add_argument("--hidden_dim",  type=int, default=128)
    p.add_argument("--n_classes",   type=int, default=10)
    p.add_argument("--n_pool",      type=int, default=512)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num_train_steps", type=int, default=200)
    p.add_argument("--data_seed",   type=int, default=0)
    p.add_argument("--train_seed",  type=int, default=7)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_id = f"{ts}_{args.baseline}_{args.budget}_gpu{args.gpu_id}"
    run_experiment(args)
