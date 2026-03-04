"""
STV (Shrinking the Variance / BLOO) estimator.

Correct BLOO implementation: baseline for prompt i = average reward of all
rollouts from OTHER prompts j != i.  The diagonal prompt-blocks of A_B are
all zero (no self-reference); each row is row-normalized by 1/((B-1)*N).

Reference: https://arxiv.org/abs/2511.03710

CUBE decomposition:
    A_B : block-structured, diagonal blocks = 0, off-diagonal = 1/((B-1)*N)
    D_B : I (identity)
    H   : H0 (uniform)
"""

import torch
from .base import BaseEstimator, RolloutBatch


class STV(BaseEstimator):
    """STV/BLOO: leave-one-prompt-out baseline.

    For prompt i, the baseline is the mean reward over all rollouts from
    prompts j != i.  Diagonal blocks of A_B are zero (no self-reference).

    CUBE decomposition:
        A_B[rows_i, cols_j] = 1/((B-1)*N)  for j != i
        A_B[rows_i, cols_i] = 0             (no self-reference)
        D_B = I
        H   = H0
    """

    def name(self) -> str:
        return "STV"

    def get_A_B(self, batch: RolloutBatch) -> torch.Tensor:
        """Return (M x M) leave-one-prompt-out baseline matrix.

        w = 1 / ((B-1) * N)
        A_B[rows_i, cols_j] = w for j != i, 0 for j == i.
        """
        B = batch.B
        N = batch.N_per_prompt
        M = B * N
        device = batch.rewards.device
        A = torch.zeros(M, M, device=device)

        if B <= 1:
            return A   # no other prompts -> zero baseline

        w = 1.0 / ((B - 1) * N)
        for i in range(B):
            rows = slice(i * N, (i + 1) * N)
            A[rows, :i * N] = w
            A[rows, (i + 1) * N:] = w

        return A

    def compute_advantage(self, batch: RolloutBatch) -> torch.Tensor:
        """Leave-one-prompt-out advantage: r_i - baseline_i.

        baseline_i = mean reward of all rollouts from prompts j != i.
        Equivalent to (I - A_B) @ r.
        """
        A_B = self.get_A_B(batch)
        baseline = A_B @ batch.rewards   # (M,)
        return batch.rewards - baseline
