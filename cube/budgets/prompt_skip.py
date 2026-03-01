"""
Prompt-skip budget module.

Skips (zeros out) a subset of prompts in each minibatch based on a score criterion.
Prompts with low reward variance or below a quality threshold are skipped.

Effect on CUBE matrices:
  H deviates from H0 by setting entire prompt blocks to zero.
  delta_H = H - H0 has nonzero blocks at skipped prompts.
  Budget bias = E[Psi (H - H0) r] is nonzero when skipped prompts correlate
  with reward signal.
"""

import torch
from .base import BaseBudget
from ..estimators.base import RolloutBatch


class PromptSkipBudget(BaseBudget):
    """Skip prompts whose reward variance is below a threshold.

    Args:
        skip_ratio: fraction of prompts to skip per minibatch (0.0 = no skip)
        criterion: 'variance' (skip low-variance) or 'random'
    """

    def __init__(self, skip_ratio: float = 0.25, criterion: str = "variance"):
        assert 0.0 <= skip_ratio < 1.0
        assert criterion in ("variance", "random")
        self.skip_ratio = skip_ratio
        self.criterion = criterion

    def name(self) -> str:
        return f"PromptSkip(ratio={self.skip_ratio}, crit={self.criterion})"

    def apply(self, batch: RolloutBatch) -> torch.Tensor:
        """Return weights: 1 for rollouts of kept prompts, 0 for skipped."""
        B = batch.rollout_counts.shape[0]
        n_skip = max(0, int(B * self.skip_ratio))
        w = torch.ones_like(batch.rewards)

        if n_skip == 0:
            return w

        if self.criterion == "variance":
            variances = torch.zeros(B, device=batch.rewards.device)
            for j in range(B):
                mask = batch.prompt_ids == j
                r_j = batch.rewards[mask]
                if r_j.numel() > 1:
                    variances[j] = r_j.var(unbiased=False)
            _, skip_idx = variances.topk(n_skip, largest=False)
        else:
            perm = torch.randperm(B, device=batch.rewards.device)
            skip_idx = perm[:n_skip]

        for j in skip_idx.tolist():
            mask = batch.prompt_ids == j
            w[mask] = 0.0

        return w
