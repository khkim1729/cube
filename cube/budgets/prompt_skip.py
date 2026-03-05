"""
Prompt-skip budget module.

Skips (zeros out) prompts whose reward variance is zero — i.e., all rollouts
for that prompt have identical rewards (all correct or all wrong).

Method: DAPO-style dynamic filtering (arXiv:2503.14476).
  DAPO filters groups where accuracy ∈ {0, 1} before gradient computation,
  as these provide no learning signal. We adopt the same criterion:
  skip prompt j if Var(r_{j,1}, ..., r_{j,N_j}) < ε.

Effect on CUBE matrices:
  H deviates from H0 by setting entire prompt blocks to zero.
  delta_H = H - H0 has nonzero blocks at skipped prompts.
  Budget bias = E[Psi (H - H0) r] is nonzero when skipped prompts correlate
  with reward signal.
"""

import torch
from .base import BaseBudget
from ..estimators.base import RolloutBatch

_VAR_EPS = 1e-8   # threshold below which reward variance is considered zero


class PromptSkipBudget(BaseBudget):
    """Skip prompts with zero reward variance (DAPO-style).

    A prompt is skipped when all its rollout rewards are identical
    (all correct or all wrong), providing no gradient signal.

    Reference: DAPO (arXiv:2503.14476), dynamic filtering of degenerate groups.
    """

    def name(self) -> str:
        return "PromptSkip(DAPO)"

    def apply(self, batch: RolloutBatch) -> torch.Tensor:
        """Return weights: 1 for active prompts, 0 for zero-variance prompts."""
        B = batch.rollout_counts.shape[0]
        w = torch.ones_like(batch.rewards)

        n_skipped = 0
        for j in range(B):
            mask = batch.prompt_ids == j
            r_j = batch.rewards[mask]
            if r_j.numel() > 1 and r_j.var(unbiased=False).item() < _VAR_EPS:
                w[mask] = 0.0
                n_skipped += 1

        # Safety: if all prompts skipped, keep all (no-op)
        if n_skipped == B:
            w.fill_(1.0)

        return w
