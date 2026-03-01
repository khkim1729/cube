"""
REINFORCE baseline estimator.

A_B = 0 (no baseline)
D_B = I (no normalization)
H   = H0 (uniform within-prompt averaging)

g_hat = (1/B) sum_j (1/Nj) sum_i psi_{j,i} * r_{j,i}
"""

import torch
from .base import BaseEstimator, RolloutBatch


class REINFORCE(BaseEstimator):
    """Plain REINFORCE without a baseline.

    Provides the simplest unbiased estimator as a reference point.
    In the CUBE notation:
        A_B = 0,  D_B = I,  H = H0
    so fusion bias and baseline bias are both zero.
    """

    def name(self) -> str:
        return "REINFORCE"

    def compute_advantage(self, batch: RolloutBatch) -> torch.Tensor:
        """Advantage = raw reward (no baseline subtraction)."""
        return batch.rewards.clone()
