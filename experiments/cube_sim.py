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
    probe_vecs: torch.Tensor,     # (R, d)
    device: str,
) -> list:
    """Compute probe projections via weighted backward passes.

    For each weight vector w, computes:
        proj[r] = v_r . ∇_θ[Σ_m w_m log π(y_m | x_m)]

    Uses ONE shared forward pass + n_w backward passes (one per weight vector).
    Formula: the weighted loss Σ_m w_m log π(y_m|x_m) has gradient = Σ_m w_m ∇_θ log π(y_m|x_m).
    (For VLMs with variable-length sequences, replace batched forward with a per-rollout loop.)

    Args:
        weight_vecs: list of n_w tensors each (M,) — e.g., [w_ghat, w_budget, ...]
        probe_vecs:  (R, d) fixed probe vectors

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
    results = []
    n_w = len(weight_vecs)
    for i, w in enumerate(weight_vecs):
        retain = (i < n_w - 1)
        loss  = (w.detach() * log_pi).sum()
        grads = torch.autograd.grad(
            loss, model.parameters(),
            retain_graph=retain, create_graph=False,
        )
        flat_g = torch.cat([g.flatten() for g in grads])  # (d,)
        results.append(probe_vecs @ flat_g)                # (R,)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rollout Sampling
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Rollouts:
    """Container for a single (X, Y) minibatch."""
    prompts: torch.Tensor       # (B, input_dim)
    golds: torch.Tensor         # (B,)
    answers: torch.Tensor       # (M,) = (B*N,) int64
    rewards: torch.Tensor       # (M,) float
    prompt_ids: torch.Tensor    # (M,) which prompt [0..B)
    N_per_prompt: int           # uniform N_j = N
    B: int
    M: int


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
        # Sample N answers per prompt
        answers_2d = torch.multinomial(
            probs.repeat_interleave(N, dim=0),               # (M, n_classes)
            num_samples=1,
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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Baseline Operators (A_B matrices)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stv_lambda(rewards: torch.Tensor, B: int, N: int) -> torch.Tensor:
    """Per-prompt James-Stein shrinkage λ_j ∈ [0,1] for STV."""
    device = rewards.device
    lambdas = torch.zeros(B, device=device)
    if B <= 1:
        return lambdas
    mu = torch.zeros(B, device=device)
    var_r = torch.zeros(B, device=device)
    for j in range(B):
        r_j = rewards[j * N:(j + 1) * N]
        mu[j] = r_j.mean()
        var_r[j] = r_j.var(unbiased=True) if N > 1 else torch.zeros(1, device=device).squeeze()
    var_mu = var_r / N
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
) -> torch.Tensor:
    """Compute A_B @ r directly via per-rollout summation (no M×M matrix).

    Avoids building the (M, M) A_B matrix entirely, saving O(M^2) memory.
    For VLMs where M can be large, this is critical.

    Returns (M,) vector = A_B r.
    """
    M = B * N
    device = rewards.device
    A_B_r = torch.zeros(M, device=device)

    if baseline == "reinforce":
        return A_B_r  # A_B = 0

    elif baseline == "grpo":
        # A_B r = group mean repeated: [mean(r_j), ..., mean(r_j)] for each j
        for j in range(B):
            s, e = j * N, (j + 1) * N
            A_B_r[s:e] = rewards[s:e].mean()

    elif baseline == "rloo":
        # A_B r = LOO mean: (sum_j - r_{j,i}) / (N-1) for each (j, i)
        if N > 1:
            for j in range(B):
                s, e = j * N, (j + 1) * N
                r_j = rewards[s:e]
                A_B_r[s:e] = (r_j.sum() - r_j) / (N - 1)

    elif baseline == "stv":
        # A_B r = (1-λ_j) * RLOO_j + λ_j * BLOO_j
        if lambdas is None:
            lambdas = _compute_stv_lambda(rewards, B, N)
        total_sum = rewards.sum()
        for j in range(B):
            s, e = j * N, (j + 1) * N
            r_j = rewards[s:e]
            lam_j = lambdas[j].item()
            # Within-prompt RLOO: (Σ r_j - r_{j,i}) / (N-1)
            rloo_j = (r_j.sum() - r_j) / (N - 1) if N > 1 else torch.zeros(N, device=device)
            # Cross-prompt BLOO: mean reward of all other prompts
            bloo_scalar = (total_sum - r_j.sum()) / ((B - 1) * N) if B > 1 else 0.0
            A_B_r[s:e] = (1 - lam_j) * rloo_j + lam_j * bloo_scalar

    return A_B_r


def build_D_B(rollouts: Rollouts, baseline: str, rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Build per-rollout normalization scale (diagonal of D_B).

    Returns diagonal entries only (M,): 1.0 for no normalization, σ^{-1} for GRPO-style.
    """
    M = rollouts.M
    B = rollouts.B
    N = rollouts.N_per_prompt
    device = rewards.device
    d = torch.ones(M, device=device)

    if baseline == "grpo":
        # Per-prompt std normalization (D_B is reward-dependent)
        for j in range(B):
            s, e = j * N, (j + 1) * N
            r_j = rewards[s:e]
            sigma = r_j.std(unbiased=False).clamp(min=eps)
            d[s:e] = 1.0 / sigma

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
        # Proportional to per-prompt reward mean (more rollouts to promising prompts)
        means = torch.tensor([
            rewards[j * N:(j + 1) * N].mean().clamp(min=0.05).item()
            for j in range(B)
        ], device=device)
        means = means / means.mean()  # normalize to mean=1

        H = torch.zeros(M, device=device)
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
) -> float:
    """Compute ||diag(H) L||_F^2 analytically (no M×M L matrix).

    L = D_B (I - A_B), so:
        ||HL||_F^2 = Σ_t (H[t] * d_B[t])^2 * ||(I-A_B)_t||^2

    Per-row squared norms of (I - A_B):
        REINFORCE:  1
        GRPO:       (N-1)/N
        RLOO:       N/(N-1)
        STV:        1 + (1-λ_j)^2/(N-1) + λ_j^2/((B-1)N)
    """
    device = rewards.device
    M = B * N
    row_sq = torch.zeros(M, device=device)

    if baseline == "reinforce":
        row_sq.fill_(1.0)

    elif baseline == "grpo":
        row_sq.fill_((N - 1) / N if N > 1 else 0.0)

    elif baseline == "rloo":
        row_sq.fill_(N / (N - 1) if N > 1 else 1.0)

    elif baseline == "stv":
        if lambdas is None:
            lambdas = _compute_stv_lambda(rewards, B, N)
        for j in range(B):
            s, e = j * N, (j + 1) * N
            lam_j = lambdas[j].item()
            # ||row_t||^2 = 1 + (1-λ_j)^2/(N-1) + λ_j^2/((B-1)N)
            val = 1.0
            if N > 1:
                val += (1 - lam_j) ** 2 / (N - 1)
            if B > 1:
                val += lam_j ** 2 / ((B - 1) * N)
            row_sq[s:e] = val

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
    probe_vecs: torch.Tensor,   # (R, d)
    S: int,
    K: int,
    B: int,
    N: int,
    probe_seed_base: int,
    device: str,
) -> CheckpointMetrics:
    """Implements plans_cube_02.txt §2 measurement protocol exactly.

    For each s in [S]: sample X_s ~ D
      For each k in [K]: sample Y_{s,k} ~ π_θ(·|X_s)
        Compute 5 projections p^1..4, q (each: (R,))
        Store HL_F^2, norms
    """
    R = probe_vecs.shape[0]
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
            rollouts = sample_rollouts(model, prompts_s, golds_s, N, rng, device)
            r = rollouts.rewards  # (M,)

            # Build operators
            d_B = build_D_B(rollouts, baseline, r)         # (M,) diagonal of D_B
            H, H0 = build_H(rollouts, budget, r)           # (M,), (M,)
            delta_H = H - H0                               # (M,)

            # Compute STV lambdas once (reused by compute_baseline_r and compute_HL_sq)
            lambdas = _compute_stv_lambda(r, B, N) if baseline == "stv" else None

            # Advantage vectors (matrix-free: direct summation, no M×M A_B)
            A_B_r   = compute_baseline_r(baseline, r, B, N, lambdas)  # (M,) = A_B r
            tilde_a = d_B * (r - A_B_r)                               # (M,) = D_B(I-A_B)r

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
                probe_vecs,         # (R, d)
                device,
            )

            p1_buf[s, k] = p1
            p2_buf[s, k] = p2
            p3_buf[s, k] = p3
            p4_buf[s, k] = p4
            q_buf[s, k]  = q

            # ── Scalar diagnostics ────────────────────────────────────────────
            # HL_F^2 = ||diag(H) L||_F^2, computed analytically (no M×M matrix)
            HL_buf[flat_idx]     = compute_HL_sq(baseline, r, B, N, H, d_B, lambdas)
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
