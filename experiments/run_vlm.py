"""
CUBE VLM Experiment Runner — single (baseline × budget) run on Qwen2-VL-7B-Instruct.

Usage:
    # Dry run (1 step, no checkpoint):
    python experiments/run_vlm.py --baseline rloo --budget none \\
        --dataset mathvista --gpu_id 1 --num_train_steps 1 --T 0 --dry_run

    # Checkpoint-only (no training):
    python experiments/run_vlm.py --baseline rloo --budget none \\
        --dataset mathvista --gpu_id 1 --num_train_steps 0 --T 1 --S 2 --K 1

    # Full run:
    python experiments/run_vlm.py --baseline rloo --budget none \\
        --dataset mathvista --gpu_id 1

CSV schema mirrors run_pilot.py + lora_rank, dataset, reward_verifiable_ratio.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.vlm_utils import (
    Rollouts,
    load_qwen_model,
    load_vlm_dataset,
    generate_rollouts_vlm,
    compute_vlm_weight_projs,
)
from experiments.cube_sim import (
    CheckpointMetrics,
    aggregate_metrics,
    compute_baseline_r,
    build_H,
    build_D_B as _build_D_B_sim,
    _compute_stv_lambda,
)


# ─────────────────────────────────────────────────────────────────────────────
# build_D_B adapted for VLM Rollouts (same logic as cube_sim.build_D_B)
# ─────────────────────────────────────────────────────────────────────────────

def build_D_B_vlm(rollouts: Rollouts, baseline: str, rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    M, B, N = rollouts.M, rollouts.B, rollouts.N_per_prompt
    d = torch.ones(M, device=rewards.device)
    if baseline == "grpo":
        for j in range(B):
            s, e = j * N, (j + 1) * N
            sigma = rewards[s:e].std(unbiased=False).clamp(min=eps)
            d[s:e] = 1.0 / sigma
    return d


# ─────────────────────────────────────────────────────────────────────────────
# CSV Schema
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    # Meta
    "run_id", "date", "time", "timestamp", "gpu_id",
    "baseline", "budget", "dataset", "lora_rank",
    # Hyperparameters
    "step", "checkpoint_idx",
    "M", "B", "N_per_prompt", "S", "K", "T", "R", "probe_seed",
    # Bias decomposition
    "total_bias_norm",
    "total_bias_proj_mean", "total_bias_proj_std",
    "budget_bias_proj_mean", "budget_bias_proj_std",
    "baseline_bias_proj_mean", "baseline_bias_proj_std",
    "fusion_bias_proj_mean", "fusion_bias_proj_std",
    # Variance decomposition
    "total_var_mean", "total_var_std",
    "within_var_mean", "within_var_std",
    "across_var_mean", "across_var_std",
    # HL Proxy
    "HL_proxy_mean", "HL_proxy_std",
    # Gradient / reward diagnostics
    "g_hat_norm_mean", "g_hat_norm_std",
    "g_ref_norm_mean", "g_ref_norm_std",
    "reward_mean", "reward_std_mean",
    "reward_verifiable_ratio",
    # Runtime
    "elapsed_seconds", "checkpoint_elapsed_seconds",
]


def write_csv_header(csv_path: Path):
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def append_csv_row(csv_path: Path, row: dict):
    with open(csv_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_checkpoint_vlm(
    model,
    processor,
    data_pool: list,        # full dataset pool
    baseline: str,
    budget: str,
    R: int,
    S: int,
    K: int,
    B: int,
    N: int,
    probe_seed: int,
    device: str,
) -> CheckpointMetrics:
    """VLM version of cube_sim.measure_checkpoint.

    S × K measurement rounds: sample B items, generate N rollouts each.
    """
    p1_buf = torch.zeros(S, K, R, device=device)
    p2_buf = torch.zeros(S, K, R, device=device)
    p3_buf = torch.zeros(S, K, R, device=device)
    p4_buf = torch.zeros(S, K, R, device=device)
    q_buf  = torch.zeros(S, K, R, device=device)

    HL_buf     = torch.zeros(S * K, device=device)
    ghat_norms = torch.zeros(S * K, device=device)
    gref_norms = torch.zeros(S * K, device=device)
    rew_means  = torch.zeros(S * K, device=device)
    rew_stds   = torch.zeros(S * K, device=device)

    flat_idx = 0

    for s in range(S):
        # Sample B items
        batch_items = random.sample(data_pool, min(B, len(data_pool)))

        for k in range(K):
            # Generate N rollouts per item
            rollouts = generate_rollouts_vlm(
                model, processor, batch_items, N,
                max_new_tokens=256, temperature=1.0, device=device,
            )
            r = rollouts.rewards  # (M,)
            M = rollouts.M

            # Build operators (same formulas as cube_sim)
            d_B = build_D_B_vlm(rollouts, baseline, r)
            H, H0 = build_H(rollouts, budget, r)
            delta_H = H - H0

            lambdas = _compute_stv_lambda(r, rollouts.B, rollouts.N_per_prompt) if baseline == "stv" else None
            A_B_r   = compute_baseline_r(baseline, r, rollouts.B, rollouts.N_per_prompt, lambdas)
            tilde_a = d_B * (r - A_B_r)

            # Weight vectors for 5 probe projections
            w1 = H * tilde_a        # g_hat
            w2 = delta_H * r        # budget bias
            w3 = H0 * A_B_r         # baseline bias
            w4 = delta_H * A_B_r    # fusion bias
            w5 = H0 * r             # g_ref

            # 5 probe projections via weighted backward passes
            model.train()  # enable grad
            p1, p2, p3, p4, q = compute_vlm_weight_projs(
                model, processor, rollouts,
                [w1, w2, w3, w4, w5],
                R=R,
                probe_seed=probe_seed,
                device=device,
            )
            model.eval()

            p1_buf[s, k] = p1
            p2_buf[s, k] = p2
            p3_buf[s, k] = p3
            p4_buf[s, k] = p4
            q_buf[s, k]  = q

            # HL_F^2 (analytical, reuse cube_sim)
            from experiments.cube_sim import compute_HL_sq
            HL_buf[flat_idx]     = compute_HL_sq(baseline, r, rollouts.B, rollouts.N_per_prompt, H, d_B, lambdas)
            ghat_norms[flat_idx] = p1.norm()
            gref_norms[flat_idx] = q.norm()
            rew_means[flat_idx]  = r.mean()
            rew_stds[flat_idx]   = r.std(unbiased=False)

            flat_idx += 1

    return CheckpointMetrics(
        p1=p1_buf, p2=p2_buf, p3=p3_buf, p4=p4_buf, q=q_buf,
        HL_proxy=HL_buf,
        g_hat_norm=ghat_norms,
        g_ref_norm=gref_norms,
        reward_mean=rew_means,
        reward_std=rew_stds,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training Step
# ─────────────────────────────────────────────────────────────────────────────

def train_step_vlm(
    model,
    processor,
    optimizer,
    data_pool: list,
    baseline: str,
    budget: str,
    B: int,
    N: int,
    device: str,
):
    """One VLM policy gradient update step."""
    batch_items = random.sample(data_pool, min(B, len(data_pool)))
    rollouts = generate_rollouts_vlm(
        model, processor, batch_items, N,
        max_new_tokens=256, temperature=1.0, device=device,
    )
    r = rollouts.rewards
    M = rollouts.M

    d_B = build_D_B_vlm(rollouts, baseline, r)
    H, H0 = build_H(rollouts, budget, r)
    lambdas = _compute_stv_lambda(r, rollouts.B, rollouts.N_per_prompt) if baseline == "stv" else None
    A_B_r   = compute_baseline_r(baseline, r, rollouts.B, rollouts.N_per_prompt, lambdas)
    tilde_a = d_B * (r - A_B_r)  # (M,) advantage

    # Compute log probs and accumulate gradients one rollout at a time (memory-efficient).
    # Avoids holding M full computation graphs simultaneously.
    from experiments.vlm_utils import compute_log_probs_batch
    weights = (H * tilde_a).detach()  # (M,) no grad needed

    optimizer.zero_grad()
    model.train()
    total_loss = 0.0
    for j, item in enumerate(rollouts.items):
        log_pi_j = compute_log_probs_batch(
            model, processor,
            [item], rollouts.responses[j * N:(j + 1) * N],
            N=N, device=device,
        )  # (N,) with grad
        loss_j = -(log_pi_j * weights[j * N:(j + 1) * N]).sum()
        loss_j.backward()
        total_loss += loss_j.item()
        torch.cuda.empty_cache()

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()

    verifiable_ratio = ((r > 0).float().mean()).item()
    return total_loss, r.mean().item(), verifiable_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)

    print(f"[{args.run_id}] GPU:{args.gpu_id}  {args.baseline} × {args.budget}  dataset={args.dataset}")
    print(f"  Device: {torch.cuda.get_device_name(args.gpu_id)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = out_dir / f"{args.run_id}.csv"
    meta_path = out_dir / f"{args.run_id}_meta.json"

    # Load model + dataset
    model, processor = load_qwen_model(lora_rank=args.lora_rank, device=device)
    data_pool = load_vlm_dataset(args.dataset, split=None, n_pool=args.n_pool)
    random.seed(args.data_seed)

    # Optimizer (LoRA params only)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    # CSV & meta
    write_csv_header(csv_path)
    meta = {
        "run_id": args.run_id,
        "baseline": args.baseline,
        "budget": args.budget,
        "dataset": args.dataset,
        "lora_rank": args.lora_rank,
        "gpu_id": args.gpu_id,
        "M": args.M, "B": args.B, "N": args.N,
        "S": args.S, "K": args.K, "T": args.T, "R": args.R,
        "probe_seed": args.probe_seed,
        "num_train_steps": args.num_train_steps,
        "lr": args.lr,
        "start_time": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if args.dry_run:
        print("  [dry_run] Skipping full loop — testing 1 training step only")
        model.train()
        loss, rew, vr = train_step_vlm(
            model, processor, optimizer, data_pool,
            args.baseline, args.budget, args.B, args.N, device,
        )
        print(f"  dry_run train step: loss={loss:.4f}, reward={rew:.3f}, verifiable_ratio={vr:.3f}")
        return csv_path

    # Training & logging loop
    log_interval = max(1, args.num_train_steps // args.T) if args.T > 0 else args.num_train_steps + 1
    total_start = time.time()
    checkpoint_idx = 0

    for step in range(args.num_train_steps + 1):
        # Checkpoint measurement
        if args.T > 0 and step % log_interval == 0 and checkpoint_idx < args.T:
            ckpt_start = time.time()
            dt = datetime.now()
            print(f"  [step={step:4d}] Measuring checkpoint {checkpoint_idx}/{args.T-1}...")

            model.eval()
            with torch.enable_grad():
                cm = measure_checkpoint_vlm(
                    model=model,
                    processor=processor,
                    data_pool=data_pool,
                    baseline=args.baseline,
                    budget=args.budget,
                    R=args.R,
                    S=args.S,
                    K=args.K,
                    B=args.B,
                    N=args.N,
                    probe_seed=args.probe_seed + step,
                    device=device,
                )
            model.train()

            metrics = aggregate_metrics(cm)
            ckpt_elapsed = time.time() - ckpt_start
            total_elapsed = time.time() - total_start

            # reward_verifiable_ratio: fraction of rollouts with reward > 0
            vr = (cm.reward_mean > 0).float().mean().item()

            row = {
                "run_id": args.run_id,
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M:%S"),
                "timestamp": dt.isoformat(),
                "gpu_id": args.gpu_id,
                "baseline": args.baseline,
                "budget": args.budget,
                "dataset": args.dataset,
                "lora_rank": args.lora_rank,
                "step": step,
                "checkpoint_idx": checkpoint_idx,
                "M": args.M, "B": args.B, "N_per_prompt": args.N,
                "S": args.S, "K": args.K, "T": args.T, "R": args.R,
                "probe_seed": args.probe_seed,
                "elapsed_seconds": round(total_elapsed, 2),
                "checkpoint_elapsed_seconds": round(ckpt_elapsed, 2),
                "reward_verifiable_ratio": round(vr, 4),
                **{k: round(v, 6) for k, v in metrics.items()},
            }
            append_csv_row(csv_path, row)

            print(
                f"    total_bias={metrics['total_bias_norm']:.4f} | "
                f"fusion_bias={metrics['fusion_bias_proj_mean']:.4f} | "
                f"HL={metrics['HL_proxy_mean']:.4f} | "
                f"reward={metrics['reward_mean']:.3f} | "
                f"vr={vr:.3f} | t={ckpt_elapsed:.1f}s"
            )
            checkpoint_idx += 1

        # Train step
        if step < args.num_train_steps:
            model.train()
            loss, rew, vr = train_step_vlm(
                model, processor, optimizer, data_pool,
                args.baseline, args.budget, args.B, args.N, device,
            )
            if step % max(1, args.num_train_steps // 20) == 0:
                print(f"  [step={step:4d}] loss={loss:.4f}  reward={rew:.3f}  vr={vr:.3f}")

    total_elapsed = time.time() - total_start
    print(f"  [{args.run_id}] Done in {total_elapsed:.1f}s → {csv_path}")

    meta["end_time"] = datetime.now().isoformat()
    meta["total_elapsed_seconds"] = round(total_elapsed, 2)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CUBE VLM experiment runner")
    # Experiment identity
    p.add_argument("--baseline",   type=str, required=True,
                   choices=["reinforce", "grpo", "rloo", "stv"])
    p.add_argument("--budget",     type=str, required=True,
                   choices=["none", "prompt_skip", "rollout_alloc", "subset_select"])
    p.add_argument("--dataset",    type=str, default="mathvista",
                   choices=["mathvista", "mmstar", "chartqa", "scienceqa", "mmbench"])
    p.add_argument("--gpu_id",     type=int, default=1)
    p.add_argument("--run_id",     type=str, default=None)
    p.add_argument("--output_dir", type=str, default="experiments/results_vlm")
    p.add_argument("--dry_run",    action="store_true")

    # VLM hyperparameters
    p.add_argument("--lora_rank",  type=int, default=16)
    p.add_argument("--n_pool",     type=int, default=512)
    p.add_argument("--M",          type=int, default=256)
    p.add_argument("--B",          type=int, default=32)
    p.add_argument("--N",          type=int, default=8)    # M = B*N
    p.add_argument("--S",          type=int, default=4)
    p.add_argument("--K",          type=int, default=2)
    p.add_argument("--T",          type=int, default=10)
    p.add_argument("--R",          type=int, default=32)
    p.add_argument("--probe_seed", type=int, default=42)

    # Training
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--num_train_steps", type=int,   default=200)
    p.add_argument("--data_seed",       type=int,   default=0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_id is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_id = f"{ts}_{args.baseline}_{args.budget}_{args.dataset}_gpu{args.gpu_id}"
    run_experiment(args)
