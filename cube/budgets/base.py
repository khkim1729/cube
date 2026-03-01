"""Base class for budget modules in CUBE.

Budget modules control how computation is allocated across prompts and rollouts.
They operate on RolloutBatch to produce a modified weight vector w and
rollout counts Nj, from which H is constructed.

Three categories (from plans_cube_01.txt):
  1. Prompt skip    - zero out entire prompts (some Nj = 0 effective)
  2. Rollout alloc  - vary Nj across prompts (non-uniform allocation)
  3. Subset select  - sample from M rollouts, set some h_m = 0
"""

from abc import ABC, abstractmethod
import torch
from ..estimators.base import RolloutBatch


class BaseBudget(ABC):
    """Abstract budget module.

    Subclasses implement apply() to return a weight tensor (M,) where
    w[m] in {0} ∪ (0, ∞). Zero weights correspond to masked rollouts.
    The budget matrix H is then H = diag(w_bar) / B where
    w_bar_{j,i} = w_{j,i} / sum_{i'} w_{j,i'}.
    """

    @abstractmethod
    def apply(self, batch: RolloutBatch) -> torch.Tensor:
        """Return per-rollout raw weights w of shape (M,)."""

    def get_H(self, batch: RolloutBatch) -> torch.Tensor:
        """Compute diagonal entries of H from raw weights.

        Returns h of shape (M,) such that H = diag(h).
        """
        w = self.apply(batch)
        B = batch.rollout_counts.shape[0]
        h = torch.zeros_like(w)
        for j in range(B):
            mask = batch.prompt_ids == j
            w_j = w[mask]
            Z_j = w_j.sum()
            if Z_j > 0:
                h[mask] = (w_j / Z_j) / B
        return h

    def get_H0(self, batch: RolloutBatch) -> torch.Tensor:
        """Reference H0 (uniform within-prompt averaging, ignoring budget)."""
        B = batch.rollout_counts.shape[0]
        h0 = torch.zeros(batch.rewards.shape[0], device=batch.rewards.device)
        for j in range(B):
            mask = batch.prompt_ids == j
            Nj = mask.sum().item()
            if Nj > 0:
                h0[mask] = 1.0 / (B * Nj)
        return h0

    @abstractmethod
    def name(self) -> str:
        """Return short identifier string."""
