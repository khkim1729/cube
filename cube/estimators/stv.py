"""
STV (Shared Token Value / batch-across) estimator.

A_B = cross-prompt blocks possible (shared across full minibatch)
D_B = I
H   = H0

The batch-level mean baseline is computed across ALL prompts and rollouts in the batch:
    b = (1/M) * sum_{j,i} r_{j,i}

This creates off-diagonal blocks in A_B, coupling prompts within the minibatch.
Per CUBE theory, cross-prompt off-diagonal routing through L = I - A_B routes
reward noise from one prompt to another, potentially amplifying variance.
"""

import torch
from .base import BaseEstimator, RolloutBatch


class STV(BaseEstimator):
    """STV: batch-across baseline (shared value across all rollouts in minibatch).

    CUBE decomposition:
        A_B : full minibatch coupling (off-diagonal blocks) → cross-prompt noise routing
        D_B : I (identity)
        H   : H0 (uniform)
    """

    def name(self) -> str:
        return "STV"

    def compute_advantage(self, batch: RolloutBatch) -> torch.Tensor:
        """Batch-mean baseline: advantage = r - mean(r over full batch)."""
        batch_mean = batch.rewards.mean()
        return batch.rewards - batch_mean

    def get_A_B(self, batch: RolloutBatch) -> torch.Tensor:
        """Return (M x M) batch-mean baseline matrix (all-ones / M).

        A_B[i, j] = 1/M for all i, j.
        """
        M = batch.rewards.shape[0]
        return torch.ones(M, M, device=batch.rewards.device) / M
