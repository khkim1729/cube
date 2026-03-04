"""
Variance measurement for CUBE gradient estimators.

Implements variance decomposition (Section 5 of the paper):

    Cov(g_hat) = E[Cov(g_hat | X)] + Cov(E[g_hat | X])
               = Within-minibatch Cov  +  Across-minibatch (prompt-mixture) Cov

The within-minibatch term is further decomposed:
    Cov(g_hat | X) = E[Cov(g_hat | X, G) | X] + Cov(E[g_hat | X, G] | X)
                   = Conditional Amplification  +  Operator Randomness

Key proxy (Theorem 2 bound):
    tr Cov(g_hat | X, G) <= ||Sigma_r||_2 * ||Psi||_2^2 * ||HL||_F^2

Lightweight proxy: ||HL||_F^2 tracks the variance trend across training steps.

Monte Carlo estimation uses:
  S: number of minibatch samples
  K: number of rollout resamples per minibatch
  R: number of probe vectors for scalarization via Tr(Cov) = E[||v^T (g-Eg)||^2]
"""

from typing import Dict
import torch


def compute_HL_proxy(
    H: torch.Tensor,   # (M,) diagonal of budget matrix
    L: torch.Tensor,   # (M, M) residual operator L = D_B(I - A_B)
) -> torch.Tensor:
    """Compute ||HL||_F^2 as the lightweight variance proxy.

    HL_F^2 = ||diag(H) @ L||_F^2 = sum_{t,m} (h_t * L_{tm})^2
    """
    HL = H.unsqueeze(1) * L   # element-wise row scaling: (M, M)
    return (HL ** 2).sum()


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
    # Here we approximate: treat each (s,k) pair as giving a sample of G
    # For each s: mean over K → this is our E[g|X,G] approximation per resample
    cond_mean_sK = G_samples.mean(dim=1)   # (S, R) ← E[g|X, realized G]
    operator_rand = cond_mean_sK.var(dim=0, unbiased=False)  # (R,)

    cond_amp = (within_total - operator_rand).clamp(min=0.0)

    return {
        "cond_amplification":  cond_amp,
        "operator_randomness": operator_rand,
    }


def compute_sourcewise_trace(
    Psi: torch.Tensor,        # (d, M) score matrix
    H: torch.Tensor,          # (M,)   diagonal of H
    L: torch.Tensor,          # (M, M) residual operator
    sigma2: torch.Tensor,     # (M,)   reward variances
) -> torch.Tensor:
    """Compute tr Cov(g|X,G) = sum_m sigma_m^2 * ||g_m||^2 (Proposition 1).

    g_m = Psi H L e_m
    """
    HL = H.unsqueeze(1) * L          # (M, M)
    G = Psi @ HL                     # (d, M)  ← G[:,m] = Psi H L e_m
    g_norms_sq = (G ** 2).sum(0)     # (M,)
    return (sigma2 * g_norms_sq).sum()
