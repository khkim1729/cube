"""
Base class for gradient estimators in the CUBE framework.

All estimators implement the unified matrix-form:
    g_hat = Psi @ H @ tilde_a
where tilde_a = D_B @ (I - A_B) @ r

Notation:
  B  : number of prompts per minibatch
  Nj : rollout count for prompt j
  M  : total rollouts = sum_j Nj
  d  : model parameter dimension (projected via probe vectors)
  Psi: (d x M) score matrix
  H  : (M x M) diagonal budget matrix
  A_B: (M x M) baseline operator
  D_B: (M x M) normalization operator
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class RolloutBatch:
    """Minibatch of rollouts for a single RL step.

    Attributes:
        rewards   : (M,)      concatenated scalar rewards
        log_probs : (M,)      per-rollout log π_θ(y|x)
        prompt_ids: (M,)      which prompt index each rollout belongs to
        rollout_counts: (B,)  Nj for each prompt (B prompts)
    """
    rewards: torch.Tensor        # (M,)
    log_probs: torch.Tensor      # (M,)
    prompt_ids: torch.Tensor     # (M,) int, values in [0, B)
    rollout_counts: torch.Tensor # (B,) int


class BaseEstimator(ABC, nn.Module):
    """Abstract gradient estimator.

    Each subclass must implement:
      - compute_advantage(batch): returns advantage vector (M,)
      - name(): human-readable identifier string

    The CUBE decomposition operators (A_B, D_B, H) can be extracted
    from estimator instances for analysis purposes.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_advantage(self, batch: RolloutBatch) -> torch.Tensor:
        """Return the advantage vector tilde_a of shape (M,)."""

    @abstractmethod
    def name(self) -> str:
        """Return a short identifier string."""

    def policy_loss(self, batch: RolloutBatch) -> torch.Tensor:
        """Standard policy gradient loss: -mean(log_prob * advantage).

        Call .backward() on the result to obtain gradients.
        """
        adv = self.compute_advantage(batch)
        return -(batch.log_probs * adv).mean()

    # ------------------------------------------------------------------
    # Helpers for CUBE analysis
    # ------------------------------------------------------------------

    def get_H(self, batch: RolloutBatch) -> torch.Tensor:
        """Return diagonal entries of budget matrix H (M,)."""
        B = batch.rollout_counts.shape[0]
        M = batch.rewards.shape[0]
        h = torch.zeros(M, device=batch.rewards.device)
        for j in range(B):
            mask = batch.prompt_ids == j
            Nj = mask.sum().item()
            if Nj > 0:
                h[mask] = 1.0 / (B * Nj)
        return h

    def get_H0(self, batch: RolloutBatch) -> torch.Tensor:
        """Return reference budget diagonal H0 (uniform per-prompt averaging)."""
        return self.get_H(batch)  # base class: H == H0 (no budget deviation)
