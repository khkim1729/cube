"""
Rollout-allocation budget module.

Allocates more rollouts to prompts considered harder or more informative.
Within each minibatch, rollout counts Nj are non-uniform.

Effect on CUBE matrices:
  H = H0 when weights are uniform within each prompt.
  However, varying Nj changes H0 itself (H0_{j,i} = 1/(B*Nj)).
  Budget bias from prompt-level allocation depends on correlation between
  Nj and the rewards of prompt j.
"""

import torch
from .base import BaseBudget
from ..estimators.base import RolloutBatch


class RolloutAllocBudget(BaseBudget):
    """Non-uniform rollout allocation per prompt.

    This module reweights rollouts proportional to allocated counts.
    If a prompt receives k_j * (M/B) rollouts (k_j > 1), its rollouts
    get proportionally higher total weight before per-prompt normalization.

    Args:
        alloc_weights: per-prompt multipliers (B,) tensor or None (uniform)
        strategy: 'proportional' (raw allocation scaling) or 'uniform'
    """

    def __init__(self, alloc_weights: torch.Tensor = None, strategy: str = "proportional"):
        self.alloc_weights = alloc_weights  # (B,) if provided
        self.strategy = strategy

    def name(self) -> str:
        return f"RolloutAlloc(strategy={self.strategy})"

    def apply(self, batch: RolloutBatch) -> torch.Tensor:
        """Return per-rollout weights reflecting allocation strategy."""
        w = torch.ones_like(batch.rewards)
        B = batch.rollout_counts.shape[0]

        if self.strategy == "uniform" or self.alloc_weights is None:
            return w

        alloc = self.alloc_weights.to(batch.rewards.device)
        alloc = alloc / alloc.mean()  # normalize to mean 1

        for j in range(B):
            mask = batch.prompt_ids == j
            if mask.any():
                w[mask] = alloc[j]

        return w
