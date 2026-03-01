"""
Bias measurement for CUBE gradient estimators.

Implements the exact global-bias decomposition (Theorem 1 in paper):

    Bias(g_hat) = E[g_hat] - nabla_theta J(theta)
               = Budget Bias  +  Baseline Bias  +  Fusion Bias

where (assuming D_B = I):
    Budget Bias   = E[Psi (H - H0) r]
    Baseline Bias = -E[Psi H0 A_B r]
    Fusion Bias   = -E[Psi (H - H0) A_B r]

In practice we estimate expectations via Monte Carlo:
  - S minibatches are sampled from D
  - K rollout sets are sampled per minibatch
  - probe vectors v_1,...,v_R are fixed globally
  - scalar projections p_{s,k,r} = <v_r, g_hat_{s,k}> are stored

All bias terms are scalarized via inner products with fixed probe vectors.
"""

from typing import Dict, List
import torch


def compute_bias(
    g_hat_samples: torch.Tensor,   # (S*K, d) or (S, K, d) gradient samples
    g_ref_samples: torch.Tensor,   # (S*K, d) reference gradient (H0, A_B=0) samples
    probe_vectors: torch.Tensor,   # (R, d) fixed probe vectors
) -> Dict[str, torch.Tensor]:
    """Estimate total bias via Monte Carlo.

    Args:
        g_hat_samples : gradient estimates from the stacked estimator
        g_ref_samples : reference gradients (REINFORCE with H0, no baseline)
        probe_vectors : fixed unit vectors for projection

    Returns:
        dict with keys: 'total_bias_proj' (R,) projected bias per probe
    """
    if g_hat_samples.dim() == 3:
        g_hat_samples = g_hat_samples.reshape(-1, g_hat_samples.shape[-1])
        g_ref_samples = g_ref_samples.reshape(-1, g_ref_samples.shape[-1])

    # E[g_hat] - E[g_ref]  (g_ref approximates nabla_theta J)
    mean_hat = g_hat_samples.mean(0)  # (d,)
    mean_ref = g_ref_samples.mean(0)  # (d,)
    total_bias = mean_hat - mean_ref  # (d,)

    # Project onto probe vectors
    bias_proj = probe_vectors @ total_bias  # (R,)

    return {"total_bias_proj": bias_proj}


def decompose_bias(
    budget_bias_samples: torch.Tensor,   # (S*K, d) Psi (H-H0) r
    baseline_bias_samples: torch.Tensor, # (S*K, d) Psi H0 A_B r
    fusion_bias_samples: torch.Tensor,   # (S*K, d) Psi (H-H0) A_B r
    probe_vectors: torch.Tensor,         # (R, d)
) -> Dict[str, torch.Tensor]:
    """Decompose bias into budget / baseline / fusion components.

    Each input tensor holds per-sample values of the corresponding bias term.
    Returns projected expectations for each component.
    """
    def proj_mean(x):
        return probe_vectors @ x.mean(0)  # (R,)

    return {
        "budget_bias_proj":   proj_mean(budget_bias_samples),
        "baseline_bias_proj": proj_mean(baseline_bias_samples),
        "fusion_bias_proj":   proj_mean(fusion_bias_samples),
    }


def compute_bias_components(
    Psi: torch.Tensor,       # (d, M) score matrix
    H: torch.Tensor,         # (M,)   diagonal of H
    H0: torch.Tensor,        # (M,)   diagonal of H0
    A_B: torch.Tensor,       # (M, M) baseline matrix
    r: torch.Tensor,         # (M,)   rewards
) -> Dict[str, torch.Tensor]:
    """Compute the three bias components for a single minibatch/rollout sample.

    Returns vectors in R^d (before probe projection).
    """
    delta_H = H - H0  # (M,)

    # Psi @ diag(delta_H) @ r
    budget_term = Psi @ (delta_H * r)          # (d,)
    # Psi @ diag(H0) @ A_B @ r
    baseline_term = Psi @ (H0 * (A_B @ r))    # (d,)
    # Psi @ diag(delta_H) @ A_B @ r
    fusion_term = Psi @ (delta_H * (A_B @ r)) # (d,)

    return {
        "budget_bias":   budget_term,
        "baseline_bias": baseline_term,
        "fusion_bias":   fusion_term,
    }
