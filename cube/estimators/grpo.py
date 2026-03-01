"""
GRPO (Group Relative Policy Optimization) estimator.

Reference: DeepSeek-R1 / GRPO paper.

A_B = prompt-block diagonal (group mean baseline)
D_B = per-prompt std normalization (reward-dependent, so introduces normalization bias)
H   = H0 (uniform within-prompt averaging)

Within each prompt j, the advantage is:
    a_{j,i} = (r_{j,i} - mean_j(r)) / std_j(r)

In CUBE notation, D_B is reward-dependent ⟹ mean-preservation is NOT guaranteed
(Eq. 4 in the paper). This is captured by the normalization bias term.
"""

import torch
from .base import BaseEstimator, RolloutBatch


class GRPO(BaseEstimator):
    """GRPO: Group Relative Policy Optimization baseline.

    Each prompt's rollouts are normalized by group mean and std.

    CUBE decomposition:
        A_B : block-diagonal with diag != 0 (group mean)
        D_B : diagonal std-normalization (reward-dependent)
        H   : H0 (uniform)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def name(self) -> str:
        return "GRPO"

    def compute_advantage(self, batch: RolloutBatch) -> torch.Tensor:
        """Per-prompt z-score normalization of rewards."""
        adv = torch.zeros_like(batch.rewards)
        B = batch.rollout_counts.shape[0]
        for j in range(B):
            mask = batch.prompt_ids == j
            r_j = batch.rewards[mask]
            if r_j.numel() <= 1:
                adv[mask] = r_j - r_j.mean()
            else:
                mu = r_j.mean()
                sigma = r_j.std(unbiased=False).clamp(min=self.eps)
                adv[mask] = (r_j - mu) / sigma
        return adv

    def get_A_B(self, batch: RolloutBatch) -> torch.Tensor:
        """Return the (M x M) baseline matrix A_B (block-diagonal group mean).

        A_B[i, j] = 1/Nk if rollouts i, j belong to the same prompt k, else 0.
        """
        M = batch.rewards.shape[0]
        A = torch.zeros(M, M, device=batch.rewards.device)
        B = batch.rollout_counts.shape[0]
        for j in range(B):
            idx = (batch.prompt_ids == j).nonzero(as_tuple=True)[0]
            Nj = len(idx)
            if Nj > 0:
                block = torch.ones(Nj, Nj, device=batch.rewards.device) / Nj
                A[idx[:, None], idx[None, :]] = block
        return A
