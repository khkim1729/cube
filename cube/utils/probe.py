"""
Probe vector management for CUBE metrics.

Instead of storing full gradient vectors (d can be billions of parameters),
we store R scalar projections p_r = <v_r, g_hat> where v_r ~ N(0, I).
This allows unbiased estimation of:
  - Bias: E[g_hat] - E[g_ref] projected onto each v_r
  - Variance: Var(<v_r, g_hat>) ≈ Tr(Cov(g_hat)) / d  (trace estimation)

The probe vectors are determined by a random seed and never stored explicitly
as large tensors on disk; only the seed is saved.
"""

import torch


def make_probe_vectors(
    param_dim: int,
    R: int = 32,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate R fixed probe vectors in R^d.

    Args:
        param_dim : model parameter dimension d
        R         : number of probe vectors
        seed      : random seed (save this, not the vectors)
        device    : torch device

    Returns:
        probes: (R, d) tensor
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(R, param_dim, generator=gen, device=device)


def project_gradient(
    grad: torch.Tensor,       # (d,) flat gradient
    probes: torch.Tensor,     # (R, d) probe vectors
) -> torch.Tensor:
    """Project gradient onto probe vectors. Returns (R,) scalar projections."""
    return probes @ grad
