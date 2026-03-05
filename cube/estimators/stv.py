"""
STV (Shrinking the Variance) estimator — full adaptive implementation.

Gradient estimator (https://arxiv.org/abs/2511.03710):

    baseline_{ij} = (1-λ_j) * μ_{j,-i}^{prompt} + λ_j * μ_{-j}^{batch}

where:
  - μ_{j,-i}^{prompt} : LOO within-prompt mean excluding rollout i
  - μ_{-j}^{batch}    : BLOO cross-prompt mean excluding prompt j
  - λ_j               : per-prompt shrinkage coefficient (data-dependent)

CUBE decomposition:
    A_B = (I - Λ) A^{prompt} + Λ A^{batch}

    A^{prompt}_j : N×N block = 1/(N-1)(11^T - I)   (within-prompt LOO = RLOO)
    A^{batch}    : BLOO matrix, zero diagonal blocks, 1/((B-1)N) off-diag
    Λ            : block-diagonal with λ_j * I_N on block j
    D_B = I,  H = H0

Shrinkage coefficient λ_j (James-Stein / empirical Bayes):
    (1) μ̂_j = (1/N) Σ_i r_{ij}
    (2) Var(μ̂_j) ≈ Var(r_{·,j}) / N
    (3) μ̄_{-j} = (1/(B-1)) Σ_{k≠j} μ̂_k
    (4) v²_j = (1/(B-1)) Σ_{k≠j} Var(μ̂_k)    ← noise: sampling variance
    (5) s²_j = (1/(B-1)) Σ_{k≠j} (μ̂_k - μ̄_{-j})²  ← signal: between-prompt
    (6) λ_j = (B-1)/B · v²_j / (v²_j + s²_j),  clipped to [0,1]
"""

import torch
from .base import BaseEstimator, RolloutBatch


def _compute_lambda(rewards: torch.Tensor, B: int, N: int) -> torch.Tensor:
    """Compute per-prompt shrinkage coefficients λ_j ∈ [0,1].

    Args:
        rewards: (M,) reward vector, ordered as B blocks of N rollouts each
        B: number of prompts
        N: rollouts per prompt (uniform)

    Returns:
        (B,) tensor of λ_j values
    """
    device = rewards.device
    lambdas = torch.zeros(B, device=device)

    if B <= 1:
        return lambdas

    # Per-prompt mean and within-prompt variance
    mu = torch.zeros(B, device=device)       # μ̂_j
    var_r = torch.zeros(B, device=device)    # Var(r_{·,j})
    for j in range(B):
        r_j = rewards[j * N:(j + 1) * N]
        mu[j] = r_j.mean()
        var_r[j] = r_j.var(unbiased=True) if N > 1 else torch.zeros(1, device=device).squeeze()

    var_mu = var_r / N   # Var(μ̂_j) ≈ Var(r_{·,j}) / N  →  (B,)

    for j in range(B):
        # Exclude prompt j
        mu_others = torch.cat([mu[:j], mu[j + 1:]])           # (B-1,)
        var_mu_others = torch.cat([var_mu[:j], var_mu[j + 1:]])  # (B-1,)

        mu_bar = mu_others.mean()                             # μ̄_{-j}
        v2 = var_mu_others.mean()                            # noise variance
        s2 = ((mu_others - mu_bar) ** 2).mean()             # signal variance

        denom = v2 + s2
        if denom > 1e-12:
            lambdas[j] = ((B - 1) / B) * v2 / denom

    return lambdas.clamp(0.0, 1.0)


class STV(BaseEstimator):
    """STV: adaptive mixture of within-prompt LOO and cross-prompt BLOO baselines.

    For prompt j, the shrinkage coefficient λ_j is computed from the data
    (James-Stein estimator): when within-prompt reward variance dominates,
    λ_j → 1 (rely on batch mean); when between-prompt variance dominates,
    λ_j → 0 (rely on within-prompt LOO).

    CUBE decomposition:
        A_B = (I - Λ) A^{prompt} + Λ A^{batch}
        D_B = I,  H = H0
    """

    def name(self) -> str:
        return "STV"

    def get_A_B(self, batch: RolloutBatch) -> torch.Tensor:
        """Return (M×M) STV baseline matrix A_B = (I-Λ)A^prompt + Λ A^batch."""
        B = batch.B
        N = batch.N_per_prompt
        M = B * N
        r = batch.rewards
        device = r.device

        # A^prompt: block-diagonal RLOO (within-prompt LOO)
        A_prompt = torch.zeros(M, M, device=device)
        if N > 1:
            for j in range(B):
                s, e = j * N, (j + 1) * N
                block = (torch.ones(N, N, device=device) - torch.eye(N, device=device)) / (N - 1)
                A_prompt[s:e, s:e] = block

        # A^batch: BLOO (leave-one-prompt-out cross-batch)
        A_batch = torch.zeros(M, M, device=device)
        if B > 1:
            w = 1.0 / ((B - 1) * N)
            for j in range(B):
                rows = slice(j * N, (j + 1) * N)
                A_batch[rows, :j * N] = w
                A_batch[rows, (j + 1) * N:] = w

        # Compute per-prompt shrinkage λ_j
        lambdas = _compute_lambda(r, B, N)   # (B,)

        # Build per-rollout λ vector (each rollout in prompt j gets λ_j)
        lam = torch.zeros(M, device=device)
        for j in range(B):
            lam[j * N:(j + 1) * N] = lambdas[j]

        # A_B[row] = (1 - λ_{row}) * A^prompt[row] + λ_{row} * A^batch[row]
        A = (1 - lam).unsqueeze(1) * A_prompt + lam.unsqueeze(1) * A_batch
        return A

    def compute_advantage(self, batch: RolloutBatch) -> torch.Tensor:
        """STV advantage: r - [(1-λ_j)*LOO_prompt + λ_j*BLOO_batch]."""
        A_B = self.get_A_B(batch)
        return batch.rewards - A_B @ batch.rewards
