"""
Utilities for constructing and manipulating RolloutBatch objects.
"""

from typing import List, Tuple
import torch
from ..estimators.base import RolloutBatch


def build_rollout_batch(
    rewards: List[List[float]],
    log_probs: List[List[float]],
    device: str = "cpu",
) -> RolloutBatch:
    """Build a RolloutBatch from nested lists.

    Args:
        rewards  : list of B lists, each containing Nj reward scalars
        log_probs: list of B lists, each containing Nj log-prob scalars

    Returns:
        RolloutBatch with concatenated tensors
    """
    B = len(rewards)
    r_flat, lp_flat, pid_flat, counts = [], [], [], []

    for j in range(B):
        Nj = len(rewards[j])
        r_flat.extend(rewards[j])
        lp_flat.extend(log_probs[j])
        pid_flat.extend([j] * Nj)
        counts.append(Nj)

    return RolloutBatch(
        rewards=torch.tensor(r_flat, dtype=torch.float32, device=device),
        log_probs=torch.tensor(lp_flat, dtype=torch.float32, device=device),
        prompt_ids=torch.tensor(pid_flat, dtype=torch.long, device=device),
        rollout_counts=torch.tensor(counts, dtype=torch.long, device=device),
    )


def concat_rollouts(batches: List[RolloutBatch]) -> RolloutBatch:
    """Concatenate multiple RolloutBatch objects along the rollout dimension.

    Prompt IDs are remapped so that prompts from batch k come after all
    prompts from batches 0..k-1.
    """
    all_r, all_lp, all_pid, all_counts = [], [], [], []
    offset = 0
    for batch in batches:
        B = batch.rollout_counts.shape[0]
        all_r.append(batch.rewards)
        all_lp.append(batch.log_probs)
        all_pid.append(batch.prompt_ids + offset)
        all_counts.append(batch.rollout_counts)
        offset += B

    return RolloutBatch(
        rewards=torch.cat(all_r),
        log_probs=torch.cat(all_lp),
        prompt_ids=torch.cat(all_pid),
        rollout_counts=torch.cat(all_counts),
    )
