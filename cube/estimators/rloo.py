"""
RLOO (Reinforce Leave-One-Out) estimator.

Reference: Ahmadian et al. "Back to Basics: Revisiting REINFORCE-Style Optimization
           for Learning from Human Feedback" (2024).

A_B = prompt-block, **diagonal-free** (self-excluding group mean)
D_B = I
H   = H0 (uniform)

The leave-one-out baseline for rollout (j,i) uses all OTHER rollouts of prompt j:
    b_{j,i} = (1/(Nj-1)) * sum_{i' != i} r_{j,i'}

Advantage:
    a_{j,i} = r_{j,i} - b_{j,i}

In CUBE notation, A_B is block-diagonal with zero diagonal entries ("self-excluding").
This property causes the baseline bias to be zero in expectation (the classic LOO result).
"""

import torch
from .base import BaseEstimator, RolloutBatch


class RLOO(BaseEstimator):
    """RLOO: Leave-One-Out baseline estimator.

    CUBE decomposition:
        A_B : prompt-block, diag = 0 (self-excluding) → baseline bias = 0
        D_B : I (identity, no normalization)
        H   : H0 (uniform)
    """

    def name(self) -> str:
        return "RLOO"

    def compute_advantage(self, batch: RolloutBatch) -> torch.Tensor:
        """Leave-one-out advantage: r_{j,i} - mean of other rollouts in prompt j."""
        adv = torch.zeros_like(batch.rewards)
        B = batch.rollout_counts.shape[0]
        for j in range(B):
            mask = batch.prompt_ids == j
            r_j = batch.rewards[mask]
            Nj = r_j.numel()
            if Nj <= 1:
                adv[mask] = torch.zeros_like(r_j)
            else:
                total = r_j.sum()
                # LOO baseline for each element: (total - r_i) / (Nj - 1)
                loo_baseline = (total - r_j) / (Nj - 1)
                adv[mask] = r_j - loo_baseline
        return adv

    def get_A_B(self, batch: RolloutBatch) -> torch.Tensor:
        """Return (M x M) self-excluding baseline matrix.

        A_B[i, j] = 1/(Nk-1) if i != j and both in prompt k, else 0.
        """
        M = batch.rewards.shape[0]
        A = torch.zeros(M, M, device=batch.rewards.device)
        B = batch.rollout_counts.shape[0]
        for j in range(B):
            idx = (batch.prompt_ids == j).nonzero(as_tuple=True)[0]
            Nj = len(idx)
            if Nj > 1:
                # all-ones block minus diagonal, divided by (Nj - 1)
                block = (torch.ones(Nj, Nj, device=batch.rewards.device)
                         - torch.eye(Nj, device=batch.rewards.device)) / (Nj - 1)
                A[idx[:, None], idx[None, :]] = block
        return A
