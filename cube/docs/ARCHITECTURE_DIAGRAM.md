# CUBE Architecture Diagram

Visual overview of the CUBE framework for beginners.
All diagrams use plain text (ASCII art).

---

## 1. The Big Picture: What CUBE Does

```
  Training Data D                    Model θ
  (questions + answers)              (ToyPolicy MLP / VLM)
         │                                  │
         ▼                                  ▼
  ┌──────────────────────────────────────────────────────┐
  │              CUBE Estimator Stack                     │
  │                                                      │
  │  Prompts X ──► Roll out N answers ──► Rewards r      │
  │                                          │            │
  │                     ┌────────────────────┤            │
  │                     │                    │            │
  │              Budget H           Baseline A_B          │
  │           (how much weight    (what baseline to       │
  │            to give each        subtract from r)       │
  │             rollout?)                   │            │
  │                     │                    │            │
  │                     └────────┬───────────┘            │
  │                              │                        │
  │                    ã = D_B (I - A_B) r                │
  │                    ĝ = Ψ H ã                          │
  │                              │                        │
  └──────────────────────────────┼───────────────────────┘
                                 │
                                 ▼
                        Gradient update: θ ← θ + α ĝ
```

---

## 2. The 4 Baseline Methods (A_B)

Each baseline defines how to compute the "subtracted baseline" from rewards:

```
Rewards per prompt (B=3 prompts, N=4 rollouts each, M=12 total):

  prompt 1: r = [1, 0, 1, 1]
  prompt 2: r = [0, 0, 1, 0]
  prompt 3: r = [1, 1, 0, 1]

REINFORCE (A_B = 0):
  A_B r = [0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0]
  (no baseline — pure policy gradient)

GRPO (group mean):
  A_B r = [0.75, 0.75, 0.75, 0.75,  0.25, 0.25, 0.25, 0.25,  0.75, 0.75, 0.75, 0.75]
  (subtract group average; also normalizes by std)

RLOO (leave-one-out mean):
  prompt 1, rollout 1: (0+1+1)/3 = 0.667
  prompt 1, rollout 2: (1+1+1)/3 = 1.000
  A_B r ≈ [0.667, 1.000, 0.667, 0.667,  ...]
  (subtract mean of OTHER rollouts in same group)

STV (adaptive mixture):
  A_B r = (1-λ_j) × RLOO_j + λ_j × BLOO_j
  BLOO_j = mean reward of all OTHER prompts
  λ_j computed via James-Stein shrinkage
  (if prompts are heterogeneous → λ≈0, RLOO dominates)
  (if prompts are homogeneous  → λ≈1, BLOO dominates)
```

---

## 3. The 4 Budget Methods (H)

Budget controls how much weight to give each rollout:

```
H_0 = 1/(B×N) = uniform weight for all rollouts (reference)

Prompt 1: r=[1,0,1,1]  (Var=0.19, mean=0.75)
Prompt 2: r=[0,0,0,0]  (Var=0.00, mean=0.00)  ← all wrong
Prompt 3: r=[0,0,0,0]  (Var=0.00, mean=0.00)  ← all wrong

None:
  H = H_0  (uniform, no change)
  All rollouts get equal weight.

PromptSkip (DAPO-style):
  Prompt 2 and 3 have Var=0 → zero weight (no learning signal)
  H = [1/4, 1/4, 1/4, 1/4,  0, 0, 0, 0,  0, 0, 0, 0]  (renormalized)

RolloutAlloc:
  Allocate more weight to high-mean prompts
  H ∝ mean(r_j)  →  more rollouts "spent" on promising prompts

SubsetSelect (top-k):
  Keep only top 2 out of 4 rollouts per prompt
  H = [1/2, 0, 1/2, 0, ...]  (only top-reward rollouts)
```

---

## 4. Bias Decomposition (Theorem 1)

```
E[ĝ] - ∇J(θ) = Total Bias
              = Budget Bias + Baseline Bias + Fusion Bias

Budget Bias   = E[ Ψ (H-H0) r ]
  ↑ non-zero when H ≠ H0 (any non-trivial budget)
  ↑ REINFORCE×any_budget always has budget bias IF budget changes H

Baseline Bias = E[ Ψ H0 A_B r ]
  ↑ non-zero when A_B ≠ 0 (any non-trivial baseline)
  ↑ RLOO: theoretically zero in expectation (LOO is unbiased)

Fusion Bias   = E[ Ψ (H-H0) A_B r ]
  ↑ non-zero ONLY WHEN BOTH budget AND baseline are active
  ↑ = 0 when H = H0 (budget=none) OR when A_B = 0 (baseline=reinforce)
  ↑ This is the NEW term discovered by CUBE

                    Budget
              None  PSkip  RAlloc  SubSel
          ┌─────┬──────┬───────┬───────┐
REINFORCE │  0  │  0   │   0   │   0   │  ← A_B=0 → always 0
     GRPO │  0  │  0   │  BIG  │ medium│  ← H+D_B interaction
     RLOO │  0  │ tiny │  BIG  │ medium│
      STV │  0  │ small│ medium│ small │  ← adaptive λ reduces it
          └─────┴──────┴───────┴───────┘
```

---

## 5. Variance Proxy ||HL||²_F (Theorem 2)

```
Theorem 2 bound:
  Tr Cov(ĝ | X, G) ≤ ||Σ_r||_2 · ||Ψ||_2^2 · ||HL||_F^2

||HL||_F^2 is the "lightweight proxy" — computed analytically:

  ||HL||_F^2 = Σ_m (h_m · d_B[m])^2 · ||(I-A_B)_m||^2
               ─────────────────────   ────────────────
               budget × normalization   baseline spread

Per-row squared norms of (I - A_B):
  REINFORCE: 1           (no subtraction → row = standard basis)
  GRPO:      (N-1)/N     (subtract group mean → compressed)
  RLOO:      N/(N-1)     (subtract LOO mean → slightly expanded)
  STV:       1 + (1-λ)²/(N-1) + λ²/((B-1)N)

GRPO disaster: d_B[m] = 1/std(r_j) can blow up if std→0
  ┌──────────────────────────────────────────────────┐
  │  GRPO HL ≈ 10^13   vs   RLOO/STV HL ≈ 10^-3     │
  │  Ratio ≈ 10^14 (!!)                              │
  └──────────────────────────────────────────────────┘
```

---

## 6. Probe Projection (Memory-Efficient)

```
Goal: estimate E[ĝ] and Var(ĝ) without storing full d-dim vectors.

Naive approach (WRONG for VLMs):
  store probe_matrix (R=32, d=100M) = 12 GB  ← impossible

CUBE approach:
  store only seed=42
  for r in 0..31:
    v_r = randn(d, seed=42+r)  ← generate on-the-fly
    p_r = dot(v_r, ĝ)          ← one scalar
  → projections = [p_0, p_1, ..., p_31]  (R=32 scalars)

Memory: O(d) per vector, not O(R×d) total.
For d=100M: 400 MB vs 12 GB.

Why is this valid?
  E[p_r^2] = E[<v_r,ĝ>^2] = ĝ^T E[v_r v_r^T] ĝ = ||ĝ||^2
  So p_r^2 is an unbiased estimator of the squared norm,
  and p_r is an unbiased estimator of <v_r, ĝ>.
```

---

## 7. Full File Map

```
cube/
├── experiments/
│   ├── cube_sim.py          ← Core engine: ToyPolicy, Rollouts, operators,
│   │                           measure_checkpoint, aggregate_metrics
│   ├── run_pilot.py         ← Single (baseline × budget) experiment
│   ├── run_batch.py         ← Multi-GPU launcher (16 combos × N runs)
│   └── analyze_results.py   ← Stats across runs per combo
│
├── cube/
│   ├── utils/
│   │   └── probe.py         ← project_flat_grad (memory-efficient probe)
│   ├── metrics/
│   │   ├── bias.py          ← Bias decomposition functions (pending update)
│   │   └── variance.py      ← Variance + HL proxy functions
│   ├── estimators/
│   │   ├── reinforce.py     ← A_B = 0
│   │   ├── grpo.py          ← A_B = group mean, D_B = 1/std
│   │   ├── rloo.py          ← A_B = LOO mean
│   │   └── stv.py           ← A_B = (1-λ)RLOO + λBLOO
│   ├── budgets/
│   │   ├── prompt_skip.py   ← Zero out Var=0 prompts
│   │   ├── rollout_alloc.py ← H ∝ mean reward
│   │   └── subset_select.py ← Top-k rollouts per prompt
│   └── docs/
│       ├── ABOUT_PILOT.md   ← Why pilot-first strategy
│       ├── PROGRESS.md      ← This pipeline explained
│       └── ARCHITECTURE_DIAGRAM.md  ← You are here
│
└── experiments/results/     ← CSV output files
    ├── <run_id>.csv
    └── combined_results.csv
```

---

## 8. The Math in One Picture

```
Given: B prompts X = {x_1,...,x_B}, N rollouts each

Step 1: Sample
  y_{j,i} ~ π_θ(·|x_j)     for j=1..B, i=1..N
  r_{j,i}  = R(y_{j,i}, x_j)   (binary: 1 if correct)

Step 2: Build operators (matrix-free)
  d_B[m]   = normalization (1.0 or 1/σ_j for GRPO)
  A_B r[m] = baseline subtraction (0, group_mean, LOO_mean, STV_blend)
  H[m]     = budget weight (uniform or skipped/reallocated)

Step 3: Compute advantage
  ã[m] = d_B[m] · (r[m] - (A_B r)[m])

Step 4: Policy gradient
  ĝ = Σ_m H[m] · ã[m] · ∇_θ log π_θ(y_m | x_m)
  (implemented as: loss = -Σ H·ã·log_π, then backward)

Step 5: Probe projection (R=32 scalars)
  for r=0..31:
    v_r ~ N(0, I_d)
    p_r = <v_r, ĝ>     ← one number summarizing gradient direction

Step 6: Measure
  Bias = E[p_r] - E[p_r^ref]   (reference: REINFORCE, H=H0)
  Variance = Var(p_r) over S×K samples
  HL_proxy = ||HL||_F^2   (analytical)
```
