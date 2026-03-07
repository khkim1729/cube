"""
Variance measurement for CUBE gradient estimators.

Implements variance decomposition (Section 5 of the paper):

    Cov(g_hat) = E[Cov(g_hat | X)] + Cov(E[g_hat | X])
               = Within-minibatch Cov  +  Across-minibatch (prompt-mixture) Cov

Key proxy (Theorem 2 bound):
    tr Cov(g_hat | X, G) <= ||Sigma_r||_2 * ||Psi||_2^2 * ||HL||_F^2

Lightweight proxy: ||HL||_F^2 tracks the variance trend across training steps.
Computed analytically (no M×M L matrix needed).

Monte Carlo estimation uses:
  S: number of minibatch samples
  K: number of rollout resamples per minibatch
  R: number of probe vectors for scalarization via Tr(Cov) = E[||v^T (g-Eg)||^2]
"""

from typing import Dict, Optional
import torch


def compute_HL_proxy(
    baseline: str,
    rewards: torch.Tensor,                    # (M,)
    B: int,
    N: int,
    H: torch.Tensor,                          # (M,) budget diagonal
    d_B: torch.Tensor,                        # (M,) D_B diagonal
    lambdas: Optional[torch.Tensor] = None,   # (B,) STV shrinkage (pre-computed)
) -> torch.Tensor:
    """Compute ||HL||_F^2 analytically (no M×M L matrix needed).

    L = D_B(I - A_B), so:
        ||HL||_F^2 = Σ_t (H[t] * d_B[t])^2 * ||(I-A_B)_t||^2

    Per-row squared norms of (I - A_B):
        REINFORCE:  1
        GRPO:       (N-1)/N
        RLOO:       N/(N-1)
        STV:        1 + (1-λ_j)^2/(N-1) + λ_j^2/((B-1)N)

    Args:
        baseline: 'reinforce', 'grpo', 'rloo', or 'stv'
        rewards:  (M,) reward vector (used only for STV lambda computation)
        B:        number of prompts
        N:        rollouts per prompt
        H:        (M,) diagonal of budget matrix
        d_B:      (M,) diagonal of D_B normalization matrix
        lambdas:  (B,) pre-computed STV lambdas (computed from rewards if None for STV)

    Returns:
        scalar tensor: ||HL||_F^2
    """
    device = rewards.device
    M = B * N
    row_sq = torch.zeros(M, device=device)

    if baseline == "reinforce":
        row_sq.fill_(1.0)

    elif baseline == "grpo":
        row_sq.fill_((N - 1) / N if N > 1 else 0.0)

    elif baseline == "rloo":
        row_sq.fill_(N / (N - 1) if N > 1 else 1.0)

    elif baseline == "stv":
        if lambdas is None:
            from cube.estimators.stv import _compute_lambda
            lambdas = _compute_lambda(rewards, B, N)
        for j in range(B):
            s, e = j * N, (j + 1) * N
            lam_j = lambdas[j].item()
            # ||row_t||^2 = 1 + (1-λ_j)^2/(N-1) + λ_j^2/((B-1)N)
            val = 1.0
            if N > 1:
                val += (1 - lam_j) ** 2 / (N - 1)
            if B > 1:
                val += lam_j ** 2 / ((B - 1) * N)
            row_sq[s:e] = val

    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    return ((H * d_B) ** 2 * row_sq).sum()


def compute_variance(
    g_hat_samples: torch.Tensor,   # (S, K, R) projected gradient samples
    # p_{s,k,r} = <v_r, g_hat_{s,k}>
) -> Dict[str, torch.Tensor]:
    """Estimate total variance via Monte Carlo projection.

    Args:
        g_hat_samples: projected gradients of shape (S, K, R)
            S = number of minibatch samples
            K = number of rollout resamples per minibatch
            R = number of probe vectors

    Returns:
        Dict with:
          'total_var'        : (R,) total variance per probe
          'within_var'       : (R,) within-minibatch variance
          'across_var'       : (R,) across-minibatch variance
    """
    S, K, R = g_hat_samples.shape

    # E[g | X_s] = mean over K rollout resamples for each minibatch s
    cond_mean = g_hat_samples.mean(dim=1)  # (S, R)

    # Within-minibatch variance: E_X[ Var_Y[g | X] ]
    # Use unbiased=True (Bessel-corrected over K samples)
    within_var = g_hat_samples.var(dim=1, unbiased=True).mean(dim=0)  # (R,)

    # Across-minibatch variance: Var_X[ E_Y[g | X] ]
    # Naive Var_S[hat_mu_s] overestimates by within_var/K:
    #   Var[hat_mu_s] = Var_X[E[g|X]] + E_X[Var[g|X]] / K
    # Bias-corrected (Bessel-corrected over S, then subtract noise floor):
    naive_across = cond_mean.var(dim=0, unbiased=True)  # (R,)
    across_var = (naive_across - within_var / K).clamp(min=0.0)

    # Total = within + across (law of total variance)
    total_var = within_var + across_var

    return {
        "total_var":  total_var,
        "within_var": within_var,
        "across_var": across_var,
    }


def decompose_variance(
    g_hat_samples: torch.Tensor,  # (S, K, R)
    G_samples: torch.Tensor,      # (S, K, R) samples with same (X, G) - for operator randomness
) -> Dict[str, torch.Tensor]:
    """Decompose within-minibatch variance into conditional amplification and operator randomness.

    Implements Eq. (9) of the paper:
        Cov(g | X) = E[Cov(g | X, G) | X] + Cov(E[g | X, G] | X)

    Args:
        g_hat_samples : (S, K, R) projected gradients
        G_samples     : same shape, used to estimate operator-randomness component

    Returns:
        Dict with:
          'cond_amplification': (R,) E[Cov(g|X,G)|X] estimate
          'operator_randomness': (R,) Cov(E[g|X,G]|X) estimate
    """
    S, K, R = g_hat_samples.shape

    # Within-minibatch total (same as compute_variance)
    within_total = g_hat_samples.var(dim=1, unbiased=False).mean(0)  # (R,)

    # Operator randomness: Var over rollout resamples of conditional mean
    cond_mean_sK = G_samples.mean(dim=1)   # (S, R)
    operator_rand = cond_mean_sK.var(dim=0, unbiased=False)  # (R,)

    cond_amp = (within_total - operator_rand).clamp(min=0.0)

    return {
        "cond_amplification":  cond_amp,
        "operator_randomness": operator_rand,
    }
