"""
Subset-selection budget module.

Sample from M rollouts and use only a subset for training (set some h_m = 0).
Common in practice for filtering low-quality or repetitive rollouts.

Effect on CUBE matrices:
  H deviates from H0 by zeroing out filtered rollouts and upweighting selected ones.
  Creates the largest delta_H among the three budget types when selection is aggressive.
  Fusion bias = E[Psi (H-H0) A_B r] can be large when selected rollouts are
  correlated with the baseline signal A_B r.
"""

import torch
from .base import BaseBudget
from ..estimators.base import RolloutBatch


class SubsetSelectBudget(BaseBudget):
    """Select top-k rollouts per prompt by reward, zeroing the rest.

    Args:
        keep_ratio: fraction of rollouts to keep per prompt (0, 1]
        criterion:  'top_reward' | 'random' | 'bottom_reward'
    """

    def __init__(self, keep_ratio: float = 0.5, criterion: str = "top_reward"):
        assert 0.0 < keep_ratio <= 1.0
        assert criterion in ("top_reward", "random", "bottom_reward")
        self.keep_ratio = keep_ratio
        self.criterion = criterion

    def name(self) -> str:
        return f"SubsetSelect(keep={self.keep_ratio}, crit={self.criterion})"

    def apply(self, batch: RolloutBatch) -> torch.Tensor:
        """Return weights: 1 for selected rollouts, 0 for filtered ones."""
        w = torch.zeros_like(batch.rewards)
        B = batch.rollout_counts.shape[0]

        for j in range(B):
            mask = batch.prompt_ids == j
            idx = mask.nonzero(as_tuple=True)[0]
            Nj = len(idx)
            if Nj == 0:
                continue

            k = max(1, int(Nj * self.keep_ratio))
            r_j = batch.rewards[idx]

            if self.criterion == "top_reward":
                _, selected = r_j.topk(k, largest=True)
            elif self.criterion == "bottom_reward":
                _, selected = r_j.topk(k, largest=False)
            else:
                perm = torch.randperm(Nj, device=batch.rewards.device)
                selected = perm[:k]

            w[idx[selected]] = 1.0

        return w
