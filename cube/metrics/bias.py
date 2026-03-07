"""
Bias measurement for CUBE gradient estimators.

Implements the bias decomposition (Theorem 1):

    Bias(g_hat) = Budget Bias + Baseline Bias + Fusion Bias

Scalarized via probe projections (no full gradient vectors stored):
    p^1_{s,k,r} = <v_r, g_hat_{s,k}>           (full estimator projection)
    p^2_{s,k,r} = <v_r, Psi (H-H0) r>          (budget bias term)
    p^3_{s,k,r} = <v_r, Psi H0 A_B r>          (baseline bias term)
    p^4_{s,k,r} = <v_r, Psi (H-H0) A_B r>      (fusion bias term)
    q_{s,k,r}   = <v_r, g_ref>                  (reference gradient)

These projections are computed via weighted backward passes in
compute_multi_weight_projs (experiments/cube_sim.py) using
project_flat_grad, which generates probe vectors one at a time from
a seed (O(d) peak memory, no (R, d) matrix ever stored in memory).
"""

from typing import Dict
import torch


def compute_bias(
    p1: torch.Tensor,  # (S, K, R) or (N, R) projected g_hat samples
    q: torch.Tensor,   # (S, K, R) or (N, R) projected g_ref samples
) -> Dict[str, torch.Tensor]:
    """Estimate total bias via Monte Carlo probe projections.

    Args:
        p1: projected g_hat samples, shape (S, K, R) or (N, R)
            p^1_{s,k,r} = <v_r, g_hat_{s,k}>
        q:  projected g_ref samples, same shape
            q_{s,k,r} = <v_r, g_ref_{s,k}>

    Returns:
        dict with 'total_bias_proj': (R,) projected bias per probe
    """
    if p1.dim() == 3:
        p1 = p1.reshape(-1, p1.shape[-1])
        q = q.reshape(-1, q.shape[-1])

    total_bias_proj = p1.mean(0) - q.mean(0)  # (R,)

    return {"total_bias_proj": total_bias_proj}


def decompose_bias(
    p2: torch.Tensor,  # (S, K, R) or (N, R) projected budget bias
    p3: torch.Tensor,  # (S, K, R) or (N, R) projected baseline bias
    p4: torch.Tensor,  # (S, K, R) or (N, R) projected fusion bias
) -> Dict[str, torch.Tensor]:
    """Decompose bias into budget / baseline / fusion components.

    Args:
        p2: projected budget bias term    <v_r, Psi (H-H0) r>
        p3: projected baseline bias term  <v_r, Psi H0 A_B r>
        p4: projected fusion bias term    <v_r, Psi (H-H0) A_B r>

    Returns:
        dict with 'budget_bias_proj', 'baseline_bias_proj', 'fusion_bias_proj': each (R,)
    """
    def proj_mean(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        return x.mean(0)  # (R,)

    return {
        "budget_bias_proj":   proj_mean(p2),
        "baseline_bias_proj": proj_mean(p3),
        "fusion_bias_proj":   proj_mean(p4),
    }
