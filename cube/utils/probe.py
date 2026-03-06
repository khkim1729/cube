"""
Probe vector management for CUBE metrics.

Instead of storing full gradient vectors (d can be billions for VLMs),
we store R scalar projections p_r = <v_r, g_hat> where v_r ~ N(0, I).
This allows unbiased estimation of:
  - Bias: E[g_hat] - E[g_ref] projected onto each v_r
  - Variance: Var(<v_r, g_hat>) ≈ Tr(Cov(g_hat)) / d  (trace estimation)

Memory design:
  - Only the seed is stored (not the vectors themselves).
  - Probe vectors are generated one at a time during projection:
      for r in range(R): v_r = randn(d); p_r = dot(v_r, g)
  - Peak extra memory: O(d), not O(R * d).
  - For VLMs with d ~ 100M LoRA params, this avoids ~12 GB of extra memory.
"""

import torch


def project_flat_grad(
    flat_grad: torch.Tensor,   # (d,) flat gradient vector
    R: int,
    seed: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Project flat gradient onto R probe vectors, one at a time.

    Generates each v_r ~ N(0, I_d) on-the-fly from a deterministic seed,
    so the full (R, d) matrix is never materialized in memory.

    Args:
        flat_grad : (d,) flattened gradient tensor
        R         : number of probe vectors
        seed      : master random seed (only this needs to be stored)
        device    : torch device

    Returns:
        projections: (R,) tensor of scalar dot products <v_r, flat_grad>
    """
    d = flat_grad.shape[0]
    projections = torch.zeros(R, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for r in range(R):
        v_r = torch.randn(d, generator=gen, device=device)
        projections[r] = (v_r * flat_grad).sum()
    return projections


def project_gradient(
    grad: torch.Tensor,       # (d,) flat gradient
    probes: torch.Tensor,     # (R, d) probe vectors
) -> torch.Tensor:
    """Project gradient onto pre-materialized probe vectors. Returns (R,) scalar projections.

    Note: materializes the full (R, d) probe matrix in memory.
    For large models (VLMs, d > 10M), use project_flat_grad instead.
    """
    return probes @ grad
