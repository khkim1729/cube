"""
CUBE Simulation Engine — plans_cube_02.txt 프로토콜 완전 구현.

측정 절차:
  고정 파라미터 θ 하에서:
    S번 미니배치 샘플링 (데이터셋 D로부터 X ~ D)
      각 미니배치에 대해 K번 롤아웃 리샘플링 (Y ~ π_θ(·|X))
        5종 벡터 계산 & R개 probe 벡터로 inner product 저장:
          p^1 = <v_r, g_hat>                  (g_hat = Psi H (I-A_B) r)
          p^2 = <v_r, Psi (H-H0) r>           (budget bias term)
          p^3 = <v_r, Psi H0 A_B r>           (baseline bias term)
          p^4 = <v_r, Psi (H-H0) A_B r>       (fusion bias term)
          q   = <v_r, g_ref> = <v_r, Psi H0 r>(reference gradient)

모델: ToyPolicy (2-layer MLP, input=64, hidden=128, output=10, d≈10K params)
보상: 이진 (정답 클래스 일치 여부, 검증 가능한 보상 설정 모사)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, functional_call

from cube.utils.probe import project_flat_grad


# ─────────────────────────────────────────────────────────────────────────────
# 1. Toy Policy Model
# ─────────────────────────────────────────────────────────────────────────────

class ToyPolicy(nn.Module):
    """Small MLP policy for CUBE simulation.

    Represents π_θ: prompt ∈ R^input_dim → distribution over n_classes answers.
    Parameters are small enough (~10K) to allow fast per-sample jacobians.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        n_classes: int = 10,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self._input_dim = input_dim
        self._n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_dim) → logits (..., n_classes)"""
        return self.fc2(torch.tanh(self.fc1(x)))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.flatten() for p in self.parameters()])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Pool
# ─────────────────────────────────────────────────────────────────────────────

class DataPool:
    """Fixed pool of (prompt, gold_answer) pairs simulating dataset D.

    prompt ∈ R^input_dim  (normalized random direction)
    gold   ∈ {0,...,n_classes-1}
    """

    def __init__(
        self,
        n_pool: int = 512,
        input_dim: int = 64,
        n_classes: int = 10,
        seed: int = 0,
        device: str = "cpu",
    ):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        prompts = torch.randn(n_pool, input_dim, generator=gen, device=device)
        prompts = prompts / prompts.norm(dim=1, keepdim=True).clamp(min=1e-8)
        golds = torch.randint(0, n_classes, (n_pool,), generator=gen, device=device)
        self.prompts = prompts
        self.golds = golds
        self.n_pool = n_pool
        self.device = device

    def sample_minibatch(self, B: int, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample B unique (prompt, gold) pairs."""
        idx = torch.randperm(self.n_pool, generator=rng, device=self.device)[:B]
        return self.prompts[idx], self.golds[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-Sample Score Vector Projection (fast via vmap)
# ─────────────────────────────────────────────────────────────────────────────

def compute_multi_weight_projs(
    model: ToyPolicy,
    prompts: torch.Tensor,        # (M, input_dim)
    answers: torch.Tensor,        # (M,) int64
    weight_vecs: list,            # list of n_w tensors, each (M,)
    R: int,                       # number of probe vectors
    probe_seed: int,              # seed for on-the-fly probe generation
    device: str,
) -> list:
    """Compute probe projections via weighted backward passes.

    For each weight vector w, computes:
        proj[r] = v_r . ∇_θ[Σ_m w_m log π(y_m | x_m)]

    Uses ONE shared forward pass + n_w backward passes (one per weight vector).
    Probe vectors v_r ~ N(0, I) are generated one at a time from probe_seed,
    so the full (R, d) matrix is never stored in memory.

    Args:
        weight_vecs: list of n_w tensors each (M,) — e.g., [w_ghat, w_budget, ...]
        R:           number of probe vectors
        probe_seed:  random seed; each call with the same seed gives the same projections

    Returns:
        list of n_w tensors each (R,) — probe projections for each weight vector
    """
    # Shared forward pass: compute log π(y_m|x_m) for all M rollouts at once.
    # Equivalent to M per-rollout forward passes (formula: ∇_θ Σ_m w_m log π(y_m|x_m)).
    # For VLMs with variable-length sequences, replace this with a per-rollout loop.
    logits    = model(prompts)                                                  # (M, C)
    log_probs = F.log_softmax(logits, dim=-1)                                   # (M, C)
    log_pi    = log_probs[torch.arange(len(answers), device=device), answers]   # (M,)

    # n_w backward passes: ∇_θ[Σ_m w_m log π(y_m|x_m)]
    # Probe vectors are generated one at a time inside project_flat_grad (O(d) peak memory).
    results = []
    n_w = len(weight_vecs)
    for i, w in enumerate(weight_vecs):
        retain = (i < n_w - 1)
        loss  = (w.detach() * log_pi).sum()
        grads = torch.autograd.grad(
            loss, model.parameters(),
            retain_graph=retain, create_graph=False,
        )
        flat_g = torch.cat([g.flatten() for g in grads])          # (d,)
        results.append(project_flat_grad(flat_g, R, probe_seed, device))  # (R,)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rollout Sampling
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Rollouts:
    """Container for a single (X, Y) minibatch."""
    prompts: torch.Tensor           # (M, input_dim)
    golds: torch.Tensor             # (M,)
    answers: torch.Tensor           # (M,) int64
    rewards: torch.Tensor           # (M,) float
    prompt_ids: torch.Tensor        # (M,) which prompt [0..B)
    N_per_prompt: int               # uniform N_j = N (or base probe N for variable)
    B: int
    M: int
    N_list: Optional[List[int]] = None  # per-prompt counts; None = uniform N_per_prompt

    def prompt_slice(self, j: int) -> slice:
        """Return flat-array slice for prompt j."""
        if self.N_list is None:
            N = self.N_per_prompt
            return slice(j * N, (j + 1) * N)
        offset = sum(self.N_list[:j])
        return slice(offset, offset + self.N_list[j])

    def prompt_N(self, j: int) -> int:
        """Return rollout count for prompt j."""
        return self.N_per_prompt if self.N_list is None else self.N_list[j]


def sample_rollouts(
    model: ToyPolicy,
    prompts: torch.Tensor,  # (B, input_dim)
    golds: torch.Tensor,    # (B,)
    N: int,
    rng: torch.Generator,
    device: str,
) -> Rollouts:
    """Sample N rollouts per prompt using π_θ. Returns Rollouts object."""
    B = prompts.shape[0]
    M = B * N

    with torch.no_grad():
        logits = model(prompts)                              # (B, n_classes)
        probs = F.softmax(logits, dim=-1)                    # (B, n_classes)
        # Sample N answers per prompt (generator=rng for reproducibility)
        answers_2d = torch.multinomial(
            probs.repeat_interleave(N, dim=0),               # (M, n_classes)
            num_samples=1,
            generator=rng,
        ).squeeze(1)                                         # (M,)
        # Rewards: 1 if answer == gold (verifiable reward)
        golds_rep = golds.repeat_interleave(N)               # (M,)
        rewards = (answers_2d == golds_rep).float()

    prompt_ids = torch.arange(B, device=device).repeat_interleave(N)  # (M,)

    return Rollouts(
        prompts=prompts.repeat_interleave(N, dim=0),  # (M, d_in)
        golds=golds_rep,
        answers=answers_2d,
        rewards=rewards,
        prompt_ids=prompt_ids,
        N_per_prompt=N,
        B=B,
        M=M,
    )


def sample_rollouts_var(
    model: ToyPolicy,
    prompts: torch.Tensor,  # (B, input_dim)
    golds: torch.Tensor,    # (B,)
    N: int,                 # base rollout count (for uniform reference)
    M: int,                 # total rollout budget
    rng: torch.Generator,
    device: str,
    n_probe: int = 4,       # probe rollouts per prompt for allocation
) -> Rollouts:
    """Sample variable N_j rollouts per prompt, allocated by reward variance.

    Implements Adaptive Rollout Allocation:
      1. Sample n_probe rollouts per prompt to estimate per-prompt accuracy.
      2. Compute weight_j = acc_j * (1 - acc_j) * 4  (Bernoulli variance proxy,
         maximized when acc_j = 0.5, i.e., prompt is near the learning boundary).
      3. Allocate N_j ∝ weight_j with sum(N_j) = M, at least 1 per prompt.
      4. Sample the actual N_j rollouts per prompt for training/measurement.
    """
    B = prompts.shape[0]

    # ── Step 1: probe rollouts ────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(prompts)                              # (B, n_classes)
        probs = F.softmax(logits, dim=-1)                    # (B, n_classes)
        probe_answers = torch.multinomial(
            probs.repeat_interleave(n_probe, dim=0),         # (B*n_probe, n_classes)
            num_samples=1, generator=rng,
        ).squeeze(1)                                         # (B*n_probe,)
        golds_probe = golds.repeat_interleave(n_probe)
        probe_rewards = (probe_answers == golds_probe).float()

    # ── Step 2: compute allocation weights ───────────────────────────────────
    weights = torch.zeros(B, device=device)
    for j in range(B):
        acc_j = probe_rewards[j * n_probe:(j + 1) * n_probe].mean()
        weights[j] = acc_j * (1.0 - acc_j) * 4.0           # Bernoulli variance × 4

    if weights.sum() < 1e-8:                                 # all prompts trivial
        weights = torch.ones(B, device=device)

    # ── Step 3: compute N_j with sum(N_j) == M, min 1 per prompt ────────────────
    weights = weights / weights.sum()
    # Guarantee at least 1 per prompt; distribute remaining M-B proportionally
    remaining = M - B
    extra = [int(round(w.item() * remaining)) for w in weights]
    diff = remaining - sum(extra)
    if diff != 0:
        sorted_idx = sorted(range(B), key=lambda j: weights[j].item(), reverse=True)
        for i in range(abs(diff)):
            j = sorted_idx[i % B]
            extra[j] = max(0, extra[j] + (1 if diff > 0 else -1))
    N_list = [1 + e for e in extra]  # guaranteed: sum == M, each >= 1
    M_actual = M  # always equals M by construction

    # ── Step 4: sample N_j actual rollouts per prompt ────────────────────────
    all_prompts_list, all_golds_list, all_answers_list = [], [], []
    all_rewards_list, all_ids_list = [], []

    with torch.no_grad():
        logits = model(prompts)                              # (B, n_classes)
        probs = F.softmax(logits, dim=-1)

        for j in range(B):
            N_j = N_list[j]
            ans_j = torch.multinomial(
                probs[j:j + 1].expand(N_j, -1),
                num_samples=1, generator=rng,
            ).squeeze(1)                                     # (N_j,)
            r_j = (ans_j == golds[j]).float()
            all_prompts_list.append(prompts[j:j + 1].expand(N_j, -1))
            all_golds_list.append(golds[j:j + 1].expand(N_j))
            all_answers_list.append(ans_j)
            all_rewards_list.append(r_j)
            all_ids_list.append(torch.full((N_j,), j, device=device))

    return Rollouts(
        prompts=torch.cat(all_prompts_list),
        golds=torch.cat(all_golds_list),
        answers=torch.cat(all_answers_list),
        rewards=torch.cat(all_rewards_list),
        prompt_ids=torch.cat(all_ids_list),
        N_per_prompt=N,         # base N (uniform reference)
        B=B,
        M=M_actual,
        N_list=N_list,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Baseline Operators (A_B matrices)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stv_lambda(
    rewards: torch.Tensor,
    B: int,
    N: int,
    rollouts: Optional["Rollouts"] = None,
) -> torch.Tensor:
    """Per-prompt James-Stein shrinkage λ_j ∈ [0,1] for STV.

    Supports variable N_j when rollouts.N_list is set.
    """
    device = rewards.device
    lambdas = torch.zeros(B, device=device)
    if B <= 1:
        return lambdas
    mu = torch.zeros(B, device=device)
    var_r = torch.zeros(B, device=device)
    for j in range(B):
        sl = rollouts.prompt_slice(j) if rollouts is not None and rollouts.N_list is not None \
             else slice(j * N, (j + 1) * N)
        N_j = rollouts.prompt_N(j) if rollouts is not None and rollouts.N_list is not None else N
        r_j = rewards[sl]
        mu[j] = r_j.mean()
        var_r[j] = r_j.var(unbiased=True) if N_j > 1 else torch.zeros(1, device=device).squeeze()
    var_mu = var_r / N   # use base N as reference (conservative)
    for j in range(B):
        mu_others = torch.cat([mu[:j], mu[j + 1:]])
        var_mu_others = torch.cat([var_mu[:j], var_mu[j + 1:]])
        mu_bar = mu_others.mean()
        v2 = var_mu_others.mean()
        s2 = ((mu_others - mu_bar) ** 2).mean()
        denom = v2 + s2
        if denom > 1e-12:
            lambdas[j] = ((B - 1) / B) * v2 / denom
    return lambdas.clamp(0.0, 1.0)


def compute_baseline_r(
    baseline: str,
    rewards: torch.Tensor,                    # (M,)
    B: int,
    N: int,
    lambdas: Optional[torch.Tensor] = None,   # (B,) STV lambdas, pre-computed
    rollouts: Optional["Rollouts"] = None,    # for variable N_j support
) -> torch.Tensor:
    """Compute A_B @ r directly via per-rollout summation (no M×M matrix).

    Avoids building the (M, M) A_B matrix entirely, saving O(M^2) memory.
    Supports variable N_j per prompt when rollouts.N_list is set.

    Returns (M,) vector = A_B r.
    """
    M = rewards.shape[0]
    device = rewards.device
    A_B_r = torch.zeros(M, device=device)

    def _sl(j):
        return rollouts.prompt_slice(j) if rollouts is not None and rollouts.N_list is not None \
               else slice(j * N, (j + 1) * N)

    def _nj(j):
        return rollouts.prompt_N(j) if rollouts is not None and rollouts.N_list is not None else N

    if baseline == "reinforce":
        return A_B_r  # A_B = 0

    elif baseline == "grpo":
        for j in range(B):
            sl = _sl(j)
            A_B_r[sl] = rewards[sl].mean()

    elif baseline == "rloo":
        for j in range(B):
            sl = _sl(j)
            N_j = _nj(j)
            r_j = rewards[sl]
            if N_j > 1:
                A_B_r[sl] = (r_j.sum() - r_j) / (N_j - 1)

    elif baseline == "stv":
        if lambdas is None:
            lambdas = _compute_stv_lambda(rewards, B, N, rollouts)
        total_sum = rewards.sum()
        for j in range(B):
            sl = _sl(j)
            N_j = _nj(j)
            r_j = rewards[sl]
            lam_j = lambdas[j].item()
            rloo_j = (r_j.sum() - r_j) / (N_j - 1) if N_j > 1 \
                     else torch.zeros(N_j, device=device)
            bloo_scalar = (total_sum - r_j.sum()) / ((B - 1) * N_j) if B > 1 else 0.0
            A_B_r[sl] = (1 - lam_j) * rloo_j + lam_j * bloo_scalar

    return A_B_r


def build_D_B(rollouts: Rollouts, baseline: str, rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Build per-rollout normalization scale (diagonal of D_B).

    Returns diagonal entries only (M,): 1.0 for no normalization, σ^{-1} for GRPO-style.
    Supports variable N_j when rollouts.N_list is set.
    """
    M = rollouts.M
    B = rollouts.B
    device = rewards.device
    d = torch.ones(M, device=device)

    if baseline == "grpo":
        for j in range(B):
            sl = rollouts.prompt_slice(j)
            r_j = rewards[sl]
            sigma = r_j.std(unbiased=False).clamp(min=eps)
            d[sl] = 1.0 / sigma

    return d  # D_B = diag(d)


def build_H(rollouts: Rollouts, budget: str, rewards: torch.Tensor, skip_ratio: float = 0.25, keep_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build budget diagonal H (M,) and reference H0 (M,).

    H0: uniform 1/(B*N_j) for all rollouts
    H:  modified by budget strategy
    """
    M = rollouts.M
    B = rollouts.B
    N = rollouts.N_per_prompt
    device = rewards.device

    # H0: uniform baseline
    H0 = torch.full((M,), 1.0 / (B * N), device=device)

    if budget == "none":
        return H0.clone(), H0

    elif budget == "prompt_skip":
        # DAPO-style: skip prompts with zero reward variance (all-correct or all-wrong).
        # Reference: DAPO (arXiv:2503.14476) — filter groups where accuracy ∈ {0, 1}.
        # These prompts provide no learning signal; including them wastes compute.
        skip_set = set()
        for j in range(B):
            r_j = rewards[j * N:(j + 1) * N]
            if r_j.var(unbiased=False).item() < 1e-8:
                skip_set.add(j)

        # Safety: if all prompts have zero variance, keep all (no skip)
        if len(skip_set) == B:
            return H0.clone(), H0

        H = H0.clone()
        for j in skip_set:
            H[j * N:(j + 1) * N] = 0.0

        return H, H0

    elif budget == "rollout_alloc":
        H = torch.zeros(M, device=device)
        if rollouts.N_list is not None:
            # Variable rollout allocation: actual N_j were sampled; H = 1/(N_j * B)
            for j in range(B):
                sl = rollouts.prompt_slice(j)
                N_j = rollouts.prompt_N(j)
                H[sl] = 1.0 / (N_j * B)
        else:
            # Uniform N: weight proportionally to per-prompt reward mean
            means = torch.tensor([
                rewards[j * N:(j + 1) * N].mean().clamp(min=0.05).item()
                for j in range(B)
            ], device=device)
            means = means / means.mean()  # normalize to mean=1
            for j in range(B):
                s, e = j * N, (j + 1) * N
                H[s:e] = (1.0 / N) * means[j] / B

        return H, H0

    elif budget == "subset_select":
        # Keep top-k rollouts per prompt by reward
        k = max(1, int(N * keep_ratio))
        w = torch.zeros(M, device=device)
        for j in range(B):
            s, e = j * N, (j + 1) * N
            r_j = rewards[s:e]
            _, top_idx = r_j.topk(k, largest=True)
            w[s + top_idx] = 1.0

        H = torch.zeros(M, device=device)
        for j in range(B):
            s, e = j * N, (j + 1) * N
            w_j = w[s:e]
            Z_j = w_j.sum()
            if Z_j > 0:
                H[s:e] = (w_j / Z_j) / B

        return H, H0

    raise ValueError(f"Unknown budget: {budget}")


def compute_HL_sq(
    baseline: str,
    rewards: torch.Tensor,                    # (M,)
    B: int,
    N: int,
    H: torch.Tensor,                          # (M,) budget diagonal
    d_B: torch.Tensor,                        # (M,) D_B diagonal
    lambdas: Optional[torch.Tensor] = None,   # (B,) STV lambdas, pre-computed
    rollouts: Optional["Rollouts"] = None,    # for variable N_j support
) -> float:
    """Compute ||diag(H) L||_F^2 analytically (no M×M L matrix).

    L = D_B (I - A_B), so:
        ||HL||_F^2 = Σ_t (H[t] * d_B[t])^2 * ||(I-A_B)_t||^2

    Per-row squared norms of (I - A_B):
        REINFORCE:  1
        GRPO:       (N_j-1)/N_j  per prompt
        RLOO:       N_j/(N_j-1)  per prompt
        STV:        1 + (1-λ_j)^2/(N_j-1) + λ_j^2/((B-1)N_j)
    Supports variable N_j when rollouts.N_list is set.
    """
    device = rewards.device
    M = rewards.shape[0]
    row_sq = torch.zeros(M, device=device)

    def _sl(j):
        return rollouts.prompt_slice(j) if rollouts is not None and rollouts.N_list is not None \
               else slice(j * N, (j + 1) * N)

    def _nj(j):
        return rollouts.prompt_N(j) if rollouts is not None and rollouts.N_list is not None else N

    if baseline == "reinforce":
        row_sq.fill_(1.0)

    elif baseline == "grpo":
        for j in range(B):
            N_j = _nj(j)
            row_sq[_sl(j)] = (N_j - 1) / N_j if N_j > 1 else 0.0

    elif baseline == "rloo":
        for j in range(B):
            N_j = _nj(j)
            row_sq[_sl(j)] = N_j / (N_j - 1) if N_j > 1 else 1.0

    elif baseline == "stv":
        if lambdas is None:
            lambdas = _compute_stv_lambda(rewards, B, N, rollouts)
        for j in range(B):
            sl = _sl(j)
            N_j = _nj(j)
            lam_j = lambdas[j].item()
            # ||row_t||^2 = 1 + (1-λ_j)^2/(N_j-1) + λ_j^2/((B-1)*N_j)
            val = 1.0
            if N_j > 1:
                val += (1 - lam_j) ** 2 / (N_j - 1)
            if B > 1:
                val += lam_j ** 2 / ((B - 1) * N_j)
            row_sq[sl] = val

    return ((H * d_B) ** 2 * row_sq).sum().item()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Core Measurement at a Fixed Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckpointMetrics:
    """All metrics at one training checkpoint (aggregated over S×K samples)."""
    # Raw probe projections: shape (S, K, R) for the 5 vectors
    p1: torch.Tensor  # g_hat projections
    p2: torch.Tensor  # budget bias term
    p3: torch.Tensor  # baseline bias term
    p4: torch.Tensor  # fusion bias term
    q:  torch.Tensor  # g_ref projections

    # Scalar arrays (S*K,)
    HL_proxy: torch.Tensor
    g_hat_norm: torch.Tensor
    g_ref_norm:  torch.Tensor
    reward_mean: torch.Tensor
    reward_std:  torch.Tensor


def measure_checkpoint(
    model: ToyPolicy,
    data_pool: DataPool,
    baseline: str,
    budget: str,
    R: int,
    probe_seed_base: int,
    S: int,
    K: int,
    B: int,
    N: int,
    device: str,
) -> CheckpointMetrics:
    """Implements plans_cube_02.txt §2 measurement protocol exactly.

    For each s in [S]: sample X_s ~ D
      For each k in [K]: sample Y_{s,k} ~ π_θ(·|X_s)
        Compute 5 projections p^1..4, q (each: (R,))
        Store HL_F^2, norms

    Probe vectors v_r ~ N(0, I) are generated one at a time from probe_seed_base
    inside compute_multi_weight_projs. The full (R, d) matrix is never stored.
    """
    M = B * N

    p1_buf = torch.zeros(S, K, R, device=device)
    p2_buf = torch.zeros(S, K, R, device=device)
    p3_buf = torch.zeros(S, K, R, device=device)
    p4_buf = torch.zeros(S, K, R, device=device)
    q_buf  = torch.zeros(S, K, R, device=device)

    HL_buf     = torch.zeros(S * K, device=device)
    ghat_norms = torch.zeros(S * K, device=device)
    gref_norms = torch.zeros(S * K, device=device)
    rew_means  = torch.zeros(S * K, device=device)
    rew_stds   = torch.zeros(S * K, device=device)

    flat_idx = 0
    rng = torch.Generator(device=device)
    rng.manual_seed(probe_seed_base + 1000)

    for s in range(S):
        # Sample minibatch X_s
        prompts_s, golds_s = data_pool.sample_minibatch(B, rng)

        for k in range(K):
            # Sample rollouts Y_{s,k}
            # rollout_alloc: sample variable N_j per prompt (actual allocation)
            if budget == "rollout_alloc":
                rollouts = sample_rollouts_var(model, prompts_s, golds_s, N, M, rng, device)
            else:
                rollouts = sample_rollouts(model, prompts_s, golds_s, N, rng, device)
            r = rollouts.rewards  # (M_actual,)

            # Build operators
            d_B = build_D_B(rollouts, baseline, r)         # (M,) diagonal of D_B
            H, H0 = build_H(rollouts, budget, r)           # (M,), (M,)
            delta_H = H - H0                               # (M,)

            # Compute STV lambdas once (reused by compute_baseline_r and compute_HL_sq)
            lambdas = _compute_stv_lambda(r, B, N, rollouts) if baseline == "stv" else None

            # Advantage vectors (matrix-free: direct summation, no M×M A_B)
            A_B_r   = compute_baseline_r(baseline, r, B, N, lambdas, rollouts)  # (M,)
            tilde_a = d_B * (r - A_B_r)                                         # (M,)

            # ── 5 weight vectors for the probe projections ────────────────────
            # w1: g_hat weights  = H * D_B * (I-A_B) * r = H * tilde_a
            w1 = H * tilde_a          # (M,)
            # w2: budget bias   = (H-H0) * r
            w2 = delta_H * r          # (M,)
            # w3: baseline bias = H0 * A_B * r
            w3 = H0 * A_B_r           # (M,)
            # w4: fusion bias   = (H-H0) * A_B * r
            w4 = delta_H * A_B_r      # (M,)
            # w5: g_ref         = H0 * r
            w5 = H0 * r               # (M,)

            # ── 5 projection vectors: 1 shared fwd + 5 weighted bwd ─────────
            p1, p2, p3, p4, q = compute_multi_weight_projs(
                model,
                rollouts.prompts,   # (M, d_in)
                rollouts.answers,   # (M,)
                [w1, w2, w3, w4, w5],
                R=R,
                probe_seed=probe_seed_base,
                device=device,
            )

            p1_buf[s, k] = p1
            p2_buf[s, k] = p2
            p3_buf[s, k] = p3
            p4_buf[s, k] = p4
            q_buf[s, k]  = q

            # ── Scalar diagnostics ────────────────────────────────────────────
            # HL_F^2 = ||diag(H) L||_F^2, computed analytically (no M×M matrix)
            HL_buf[flat_idx]     = compute_HL_sq(baseline, r, B, N, H, d_B, lambdas, rollouts)
            ghat_norms[flat_idx] = p1.norm()
            gref_norms[flat_idx] = q.norm()
            rew_means[flat_idx]  = r.mean()
            rew_stds[flat_idx]   = r.std(unbiased=False)

            flat_idx += 1

    return CheckpointMetrics(
        p1=p1_buf, p2=p2_buf, p3=p3_buf, p4=p4_buf, q=q_buf,
        HL_proxy=HL_buf,
        g_hat_norm=ghat_norms,
        g_ref_norm=gref_norms,
        reward_mean=rew_means,
        reward_std=rew_stds,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metric Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_metrics(cm: CheckpointMetrics) -> Dict[str, float]:
    """Aggregate CheckpointMetrics into scalar summary statistics.

    Implements the bias/variance decomposition from the paper.
    """
    S, K, R = cm.p1.shape

    # ── Bias ─────────────────────────────────────────────────────────────────
    mean_p1 = cm.p1.reshape(-1, R).mean(0)   # E[g_hat]   (R,)
    mean_q  = cm.q.reshape(-1, R).mean(0)    # E[g_ref]   (R,)
    total_bias_proj = mean_p1 - mean_q        # (R,)

    mean_p2 = cm.p2.reshape(-1, R).mean(0)   # E[budget_term]
    mean_p3 = cm.p3.reshape(-1, R).mean(0)   # E[baseline_term]
    mean_p4 = cm.p4.reshape(-1, R).mean(0)   # E[fusion_term]

    # ── Variance (Law of Total Variance over S, K) ────────────────────────────
    # Within-minibatch: E_X[Var_Y[g|X]] = mean over S of Var over K
    # Use Bessel-corrected (unbiased=True) variance over K samples
    within_var_per_probe = cm.p1.var(dim=1, unbiased=True).mean(0)  # (R,)

    # Across-minibatch: Var_X[E_Y[g|X]]
    # Naive Var_S[hat_mu_s] overestimates by within/K (noise from K-sample mean).
    # Bias-corrected: across = Var_S[hat_mu_s] - within / K
    cond_mean = cm.p1.mean(dim=1)             # (S, R)
    naive_across = cond_mean.var(dim=0, unbiased=True)  # (R,) Bessel-corrected
    across_var_per_probe = (naive_across - within_var_per_probe / K).clamp(min=0.0)

    # Total = within + across (law of total variance)
    total_var_per_probe = within_var_per_probe + across_var_per_probe

    def s(t): return t.mean().item()  # scalar mean
    def se(t): return t.std(unbiased=False).item()  # scalar std

    return {
        # ── Bias ───────────────────────────────────────────────────────────
        "total_bias_norm":           total_bias_proj.norm().item(),
        "total_bias_proj_mean":      total_bias_proj.abs().mean().item(),
        "total_bias_proj_std":       total_bias_proj.std(unbiased=False).item(),
        "budget_bias_proj_mean":     mean_p2.abs().mean().item(),
        "budget_bias_proj_std":      mean_p2.std(unbiased=False).item(),
        "baseline_bias_proj_mean":   mean_p3.abs().mean().item(),
        "baseline_bias_proj_std":    mean_p3.std(unbiased=False).item(),
        "fusion_bias_proj_mean":     mean_p4.abs().mean().item(),
        "fusion_bias_proj_std":      mean_p4.std(unbiased=False).item(),
        # ── Variance ───────────────────────────────────────────────────────
        "total_var_mean":            s(total_var_per_probe),
        "total_var_std":             se(total_var_per_probe),
        "within_var_mean":           s(within_var_per_probe),
        "within_var_std":            se(within_var_per_probe),
        "across_var_mean":           s(across_var_per_probe),
        "across_var_std":            se(across_var_per_probe),
        # ── HL Proxy ───────────────────────────────────────────────────────
        "HL_proxy_mean":             s(cm.HL_proxy),
        "HL_proxy_std":              se(cm.HL_proxy),
        # ── Gradient / Reward Diagnostics ──────────────────────────────────
        "g_hat_norm_mean":           s(cm.g_hat_norm),
        "g_hat_norm_std":            se(cm.g_hat_norm),
        "g_ref_norm_mean":           s(cm.g_ref_norm),
        "g_ref_norm_std":            se(cm.g_ref_norm),
        "reward_mean":               s(cm.reward_mean),
        "reward_std_mean":           s(cm.reward_std),
    }
