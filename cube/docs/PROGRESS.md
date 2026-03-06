# CUBE Experiment Pipeline — Step-by-Step Guide

This document explains how a CUBE experiment actually runs, end-to-end.
It maps each step to the corresponding Python code, the data flow,
and the mathematical expression from the paper.

---

## 1. Entry Point: `run_pilot.py`

```bash
python experiments/run_pilot.py \
    --baseline rloo --budget none \
    --gpu_id 1 --S 4 --K 2 --T 10
```

This is the main script for a single `(baseline × budget)` combination.
It orchestrates the training loop and measurement checkpoints.

**What it does:**
1. Creates a `ToyPolicy` model and `DataPool`
2. Trains with policy gradient for `num_train_steps` steps
3. At `T` evenly-spaced checkpoints, calls `measure_checkpoint()`
4. Writes results to a CSV file

---

## 2. The Model: `ToyPolicy` (`cube_sim.py`, line 33)

```python
class ToyPolicy(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, n_classes=10):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))
```

**What this represents in the paper:**
The policy `π_θ(a | x)` — a distribution over answers given a prompt.
In a real VLM, this would be the language model generating token sequences.
Here it is a small MLP for fast simulation.

**Parameters:** d = 9,610 scalars (64×128 + 128 + 128×10 + 10)

---

## 3. Data: `DataPool` (`cube_sim.py`, line 70)

```python
data_pool = DataPool(n_pool=512, input_dim=64, n_classes=10, seed=0)
```

Simulates a dataset `D` of question-answer pairs:
- `prompts[i] ∈ R^64`: normalized random vector (simulates a question embedding)
- `golds[i] ∈ {0,...,9}`: correct answer label

**Paper notation:** `x ~ D` (sampling from dataset distribution)

---

## 4. Rollout Sampling: `sample_rollouts()` (`cube_sim.py`, line 169)

```python
rollouts = sample_rollouts(model, prompts_b, golds_b, N=8, rng, device)
```

For B=32 prompts, samples N=8 answers each → M=256 total rollouts.

**Output:**
- `rollouts.answers` — (M,) predicted answers, `y_m ~ π_θ(·|x_j)`
- `rollouts.rewards` — (M,) binary reward: `r_m = 1[y_m == gold_j]`

**Paper notation:**
```
Y = {y_{j,i}}  for j=1..B, i=1..N
r_m = R(y_m, x_m)  (verifiable reward)
```

---

## 5. Building the CUBE Operators

### 5a. Normalization `D_B`: `build_D_B()` (`cube_sim.py`, line 289)

```python
d_B = build_D_B(rollouts, baseline="grpo", rewards)
# d_B[j*N:(j+1)*N] = 1 / std(r_j)  for GRPO
# d_B = ones(M)                       for others
```

**Paper notation:** `D_B = diag(d_B)`, reward-dependent for GRPO.

### 5b. Baseline operator `A_B`: `compute_baseline_r()` (`cube_sim.py`, line 236)

```python
A_B_r = compute_baseline_r(baseline="rloo", rewards, B=32, N=8)
# A_B_r[j*N + i] = (sum(r_j) - r_{j,i}) / (N-1)  for RLOO (LOO mean)
# A_B_r = 0                                         for REINFORCE
# A_B_r[j*N:(j+1)*N] = mean(r_j)                  for GRPO
```

This computes `A_B @ r` directly (no M×M matrix stored):

| Baseline | `A_B @ r` formula |
|----------|-------------------|
| REINFORCE | `0` |
| GRPO | group mean `μ_j` repeated N times |
| RLOO | LOO mean `(Σ r_j - r_{j,i}) / (N-1)` |
| STV | `(1-λ_j) × RLOO_j + λ_j × BLOO_j` |

**Paper notation:** `A_B r` where `A_B` encodes the baseline operation.

### 5c. Budget matrix `H`: `build_H()` (`cube_sim.py`, line 311)

```python
H, H0 = build_H(rollouts, budget="prompt_skip", rewards)
# H0 = 1/(B*N) for all m  (uniform reference)
# H  = 0 for prompts with Var(r_j) = 0  (DAPO-style PromptSkip)
```

**Paper notation:** `H = diag(h_m)`, `H_0 = (1/M) I` (uniform reference).

### 5d. Advantage: `tilde_a`

```python
tilde_a = d_B * (rewards - A_B_r)   # (M,)
```

**Paper notation:** `ã = D_B (I - A_B) r`

---

## 6. The Gradient Estimator

The training loss at each step:

```python
# run_pilot.py, train_step()
loss = -(lp_selected * (H * tilde_a)).sum()
```

**Paper notation (main formula):**
```
ĝ = Ψ H D_B (I - A_B) r
```

where `Ψ_{d×M}` has column `∇_θ log π_θ(y_m | x_m)`.
The loss is the negative of this, i.e. `-Σ_m h_m ã_m log π(y_m|x_m)`.

---

## 7. Checkpoint Measurement: `measure_checkpoint()` (`cube_sim.py`, line 458)

This is the core of the CUBE evaluation protocol (Theorem 1 + 2).

```python
cm = measure_checkpoint(
    model, data_pool,
    baseline="rloo", budget="none",
    R=32, probe_seed_base=42,
    S=4, K=2, B=32, N=8, device="cuda:1"
)
```

**What it does (S×K loops):**

```
For s = 1..S:
    Sample B prompts from D   →  X_s
    For k = 1..K:
        Sample N rollouts per prompt  →  Y_{s,k}
        Compute rewards r (M,)
        Compute operators: d_B, H, H0, A_B_r
        Compute 5 weight vectors w1..w5
        Do 1 forward + 5 backward passes → 5 gradient vectors
        Project each onto R probe vectors → 5 × (R,) scalars
        Store HL_F^2, ||ĝ||, rewards
```

**The 5 weight vectors** (from Theorem 1 bias decomposition):

| Weight | Formula | What it measures |
|--------|---------|-----------------|
| `w1 = H * tilde_a` | `H D_B (I-A_B) r` | `ĝ` itself |
| `w2 = (H-H0) * r` | `(H-H_0) r` | Budget bias term |
| `w3 = H0 * A_B_r` | `H_0 A_B r` | Baseline bias term |
| `w4 = (H-H0) * A_B_r` | `(H-H_0) A_B r` | Fusion bias term |
| `w5 = H0 * r` | `H_0 r` | Reference gradient `g_ref` |

**Why 5 backward passes instead of 5 separate forward passes?**

Each weight vector `w_i` gives:
```
∇_θ [Σ_m w_i[m] · log π(y_m|x_m)] = Σ_m w_i[m] · ∇_θ log π(y_m|x_m) = Ψ w_i
```

So one shared forward pass (computing all `log π(y_m|x_m)`) is enough,
and 5 backward passes give 5 different `Ψ w_i` gradient vectors.

---

## 8. Probe Projection: `project_flat_grad()` (`cube/utils/probe.py`)

```python
proj = project_flat_grad(flat_g, R=32, seed=42, device="cuda")
# Returns: (R,) tensor where proj[r] = <v_r, flat_g>
# v_r ~ N(0, I_d) generated on-the-fly from seed (not stored)
```

**Why not store the full (R, d) probe matrix?**

For ToyPolicy: d=9,610, R=32 → 307K floats ≈ 1.2 MB (fine).
For VLMs: d≈100M LoRA params, R=32 → 3.2B floats ≈ 12 GB (impossible).

So we generate each `v_r` one at a time, compute the dot product, then discard.

**Paper notation:** `p_r = <v_r, ĝ>` where `v_r ~ N(0, I_d)` are fixed probes.

---

## 9. Metric Aggregation: `aggregate_metrics()` (`cube_sim.py`, line 570)

```python
metrics = aggregate_metrics(cm)
# metrics["total_bias_norm"]       = ||E[ĝ] - E[g_ref]||
# metrics["fusion_bias_proj_mean"] = |E[(H-H0) A_B r projections]|
# metrics["HL_proxy_mean"]         = ||HL||_F^2
# metrics["total_var_mean"]        = Var(ĝ projections)
```

**Bias (Theorem 1):**
```
Bias(ĝ) = E[ĝ] - ∇J(θ)
         = E[Ψ(H-H0)r]        (budget bias)
         - E[Ψ H0 A_B r]      (baseline bias)
         - E[Ψ(H-H0) A_B r]   (fusion bias)
```

Estimated via: `E[proj_r(ĝ)] - E[proj_r(g_ref)]` averaged over S×K samples.

**Variance (Theorem 2):**
```
Var(ĝ) = E_X[Var_Y[ĝ|X]]    (within-minibatch)
        + Var_X[E_Y[ĝ|X]]    (across-minibatch)
```

Bias-corrected across estimate:
```
across_var = Var_S[hat_μ_s] - within_var / K
```

**HL proxy:**
```
||HL||_F^2 = Σ_m (h_m · d_B[m])^2 · ||(I-A_B)_m||^2
```

Computed analytically without materializing the M×M matrix `L = D_B(I-A_B)`.

---

## 10. Output: CSV File

Each checkpoint measurement writes one row to `experiments/results/<run_id>.csv`.

Example columns:

| Column | Value | Meaning |
|--------|-------|---------|
| `step` | 100 | Training step |
| `total_bias_norm` | 0.0034 | `||E[ĝ] - E[g_ref]||` |
| `fusion_bias_proj_mean` | 0.0000 | Fusion bias (0 if `H=H0` or `A_B=0`) |
| `HL_proxy_mean` | 4.46e-3 | Variance proxy `||HL||_F^2` |
| `reward_mean` | 0.125 | Average reward across S×K rollouts |

---

## 11. Multi-Run Launcher: `run_batch.py`

```bash
CUDA_VISIBLE_DEVICES=1 python3 experiments/run_batch.py \
    --gpu_id 1 --n_runs 15 --offset 0 --S 4 --K 2
```

Runs 15 experiments sequentially on GPU 1, cycling through the 16 `(baseline × budget)` combinations starting at offset 0.

**16 combinations = 4 baselines × 4 budgets:**

```
[REINFORCE, GRPO, RLOO, STV] × [none, prompt_skip, rollout_alloc, subset_select]
```

---

## Summary: Data Flow

```
DataPool (synthetic dataset D)
    ↓
sample_minibatch(B=32)   → B prompts + golds
    ↓
sample_rollouts(N=8)     → M=256 (answers, rewards)
    ↓
build_D_B, build_H, compute_baseline_r
    → operators: d_B (M,), H (M,), A_B_r (M,)
    ↓
weight vectors w1..w5    → 5 × (M,) vectors
    ↓
1 forward + 5 backward   → 5 × (d,) gradient vectors
    ↓
project_flat_grad × 5    → 5 × (R,) probe projections  [no (R,d) matrix!]
    ↓
aggregate_metrics        → bias_norm, fusion_bias, HL_proxy, variance
    ↓
CSV row
```
