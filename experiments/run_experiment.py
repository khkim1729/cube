"""
CUBE experiment runner.

Runs bias/variance analysis experiments for all combinations of:
  - Baseline methods: REINFORCE, GRPO, RLOO, STV
  - Budget methods:   PromptSkip, RolloutAlloc, SubsetSelect (+ no-budget)

For each combination, the runner:
  1. Trains or loads a VLM policy
  2. At T logging checkpoints, freezes parameters
  3. Samples S minibatches, each resampled K times
  4. Computes and stores bias / variance metrics via probe projections
  5. Saves results to experiments/<model>/<datetime>/

Usage:
    python experiments/run_experiment.py --config configs/default.yaml

Output directory structure:
    experiments/
      <model_name>/
        <YYYYMMDD_HHMMSS>/
          config.yaml
          metrics/
            bias_heatmap.json
            variance_heatmap.json
            per_step_*.json
          checkpoints/    (gitignored)
          logs/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cube.estimators import REINFORCE, GRPO, RLOO, STV
from cube.budgets import PromptSkipBudget, RolloutAllocBudget, SubsetSelectBudget
from cube.metrics import (
    compute_bias, decompose_bias, compute_bias_components,
    compute_variance, decompose_variance, compute_HL_proxy,
)
from cube.utils import project_flat_grad


# Default hyperparameters (from plans_cube_02.txt)
DEFAULT_CONFIG = {
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "dataset": "mathvista",
    "dataset_split": "testmini",
    "n_baselines": 4,    # number of baseline methods
    "m_budgets": 3,      # number of budget methods
    "M": 256,            # total rollouts per minibatch
    "B": 32,             # prompts per minibatch
    "S": 16,             # minibatch sampling repetitions (logging)
    "K": 8,              # rollout resamples per minibatch (logging)
    "T": 10,             # logging checkpoints during training
    "R": 32,             # probe vector count
    "probe_seed": 42,
    "max_new_tokens": 256,
    "temperature": 1.0,
    "learning_rate": 1e-5,
    "num_train_steps": 200,
    "device": "cuda",
    "output_root": "experiments",
}

BASELINES = {
    "reinforce": REINFORCE,
    "grpo": GRPO,
    "rloo": RLOO,
    "stv": STV,
}

BUDGETS = {
    "none": None,
    "prompt_skip": lambda: PromptSkipBudget(skip_ratio=0.25),
    "rollout_alloc": lambda: RolloutAllocBudget(strategy="proportional"),
    "subset_select": lambda: SubsetSelectBudget(keep_ratio=0.5),
}


def make_run_dir(output_root: str, model_name: str) -> Path:
    """Create timestamped experiment directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.replace("/", "_").replace("-", "_")
    run_dir = Path(output_root) / model_short / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def save_config(run_dir: Path, config: dict):
    """Save experiment config to run directory."""
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_bias_variance_sweep(config: dict, run_dir: Path):
    """Run bias/variance measurement sweep for all baseline x budget combinations.

    For each combo, simulate S x K gradient samples and compute metrics.
    Results are saved as JSON files in run_dir/metrics/.
    """
    R = config["R"]
    S = config["S"]
    K = config["K"]
    M = config["M"]
    B = config["B"]
    probe_seed = config["probe_seed"]
    # Probe vectors are NOT materialized as an (R, d) matrix.
    # Instead, project_flat_grad(flat_g, R, seed, device) generates each
    # v_r ~ N(0, I_d) on-the-fly and accumulates the dot product.
    # This keeps peak extra memory at O(d) instead of O(R * d).

    results = {}

    for bl_name, bl_cls in BASELINES.items():
        for budget_name, budget_fn in BUDGETS.items():
            key = f"{bl_name}_x_{budget_name}"
            print(f"  Running combo: {key}")

            # Simulate gradient samples (placeholder for real training loop).
            # In the real implementation, compute flat_g from backward passes
            # and project via: proj = project_flat_grad(flat_g, R, probe_seed, device)
            # This avoids storing the full (R, d) probe matrix.
            torch.manual_seed(hash(key) % (2**31))
            g_hat_samples = torch.randn(S, K, R)  # projected gradients (placeholder)
            g_ref_samples = torch.randn(S * K, R)

            # Bias metrics
            bias_metrics = compute_bias(
                g_hat_samples.reshape(-1, R),
                g_ref_samples,
                torch.eye(R),  # probes = identity when already projected
            )

            # Variance metrics
            var_metrics = compute_variance(g_hat_samples)

            # HL proxy
            L = torch.eye(M) - torch.ones(M, M) / M  # placeholder STV L
            H = torch.ones(M) / (B * (M // B))
            hl_proxy = compute_HL_proxy(H, L).item()

            results[key] = {
                "total_bias_norm": bias_metrics["total_bias_proj"].norm().item(),
                "total_var": var_metrics["total_var"].mean().item(),
                "within_var": var_metrics["within_var"].mean().item(),
                "across_var": var_metrics["across_var"].mean().item(),
                "HL_proxy": hl_proxy,
            }

    # Save heatmap-ready summary
    heatmap_path = run_dir / "metrics" / "bias_variance_heatmap.json"
    with open(heatmap_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved metrics to {heatmap_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run CUBE bias/variance experiments")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="experiments")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run metric computation only (no model training)")
    args = parser.parse_args()

    # Load config
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config.update(yaml.safe_load(f))
    if args.model:
        config["model"] = args.model
    if args.dataset:
        config["dataset"] = args.dataset
    if args.output_root:
        config["output_root"] = args.output_root

    # Create run directory
    run_dir = make_run_dir(config["output_root"], config["model"])
    save_config(run_dir, config)
    print(f"Run directory: {run_dir}")

    # Run sweep
    print("Running bias/variance sweep...")
    results = run_bias_variance_sweep(config, run_dir)

    print("\nSummary:")
    for key, vals in results.items():
        print(f"  {key:40s}  bias={vals['total_bias_norm']:.4f}  "
              f"var={vals['total_var']:.4f}  HL={vals['HL_proxy']:.4f}")

    print(f"\nDone. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
