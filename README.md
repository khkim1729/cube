# CUBE: Coupled Updates and Budget Estimators in Reinforcement Learning

> **[한국어 README](README_KR.md)** | **English**

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Framework Architecture](#framework-architecture)
- [Figures](#figures)
- [Code Structure](#code-structure)
- [Baselines & Budget Methods](#baselines--budget-methods)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Citation](#citation)

---

## Overview

**CUBE** (**C**oupled **U**pdates and **B**udget **E**stimators) is a unified theoretical and empirical framework for analyzing gradient estimators in Reinforcement Learning from Verifiable Rewards (RLVR), with a focus on Vision-Language Models (VLMs).

Modern RLVR pipelines combine:
- **Baseline methods** (REINFORCE, GRPO, RLOO, STV) to reduce variance
- **Budget modules** (prompt skipping, rollout allocation, subset selection) to control compute

CUBE expresses the entire estimator in a **matrix form**:

$$\hat{g} = \Psi H \tilde{a} = \Psi H D_B (I - A_B) r$$

where:
| Symbol | Meaning |
|--------|---------|
| $\Psi \in \mathbb{R}^{d \times M}$ | Per-rollout score/feature matrix |
| $H \in \mathbb{R}^{M \times M}$ | Diagonal budget matrix |
| $A_B \in \mathbb{R}^{M \times M}$ | Baseline operator (may couple prompts) |
| $D_B \in \mathbb{R}^{M \times M}$ | Normalization operator |
| $r \in \mathbb{R}^M$ | Concatenated reward vector |

---

## Key Contributions

### 1. Exact Global Bias Decomposition (Theorem 1)

$$\mathbb{E}[\hat{g}] - \nabla_\theta J(\theta) = \underbrace{\mathbb{E}[\Psi(H-H_0)r]}_{\text{Budget Bias}} - \underbrace{\mathbb{E}[\Psi H_0 A_B r]}_{\text{Baseline Bias}} - \underbrace{\mathbb{E}[\Psi(H-H_0)A_B r]}_{\text{Fusion Bias}}$$

The **Fusion Bias** is a cross-term that arises only when budget and baseline are **both** active — it cannot be detected by ablating either module alone.

### 2. Noise Amplification Bound (Theorem 2)

$$\mathrm{tr}\, \mathrm{Cov}(\hat{g} \mid X, G) \leq \|\Sigma_r\|_2 \cdot \|\Psi\|_2^2 \cdot \|HL\|_F^2$$

The proxy $\|HL\|_F^2$ tracks variance amplification through routing ($L$) and weight concentration ($H$).

### 3. Source-Wise Variance Propagation (Proposition 1)

$$\mathrm{tr}\, \mathrm{Cov}(\hat{g} \mid X, G) = \sum_{m=1}^M \sigma_m^2 \|g_m\|_2^2, \quad g_m = \Psi H L e_m$$

Each coordinate noise source $\sigma_m^2$ contributes a weighted term governed by routing and budget amplification.

---

## Figures

### Figure 1: CUBE Framework Overview

<p align="center">
  <img src="assets/fig1.png" width="800" alt="CUBE Framework Overview"/>
</p>

**Figure 1.** Overview of the CUBE framework. The central pipeline shows the unified matrix-form Monte Carlo estimator. The left panel shows the exact global bias decomposition (Theorem 1): Total Bias = Budget Bias + Baseline Bias + Fusion Bias. The right panel shows variance amplification under minibatch coupling (Theorem 2), highlighting how off-diagonal noise routing and weight concentration amplify conditional variance.

---

### Figure 2: VLM-RL Diagnosis and Mitigation

<p align="center">
  <img src="assets/fig2.png" width="800" alt="VLM-RL Diagnosis with CUBE"/>
</p>

**Figure 2.** Diagnosing and mitigating RL instability in VLMs using CUBE. The left panel shows standard VLM-RL training where coupled stacking induces high variance and training divergence. The right panel applies CUBE's matrix-form analysis to decompose bias and trace source-wise variance propagation, enabling stable convergence on multimodal reasoning benchmarks.

---

## Code Structure

```
cube/
├── cube/
│   ├── estimators/          # Baseline gradient estimators
│   │   ├── base.py          # Abstract BaseEstimator, RolloutBatch
│   │   ├── reinforce.py     # REINFORCE (A_B=0, D_B=I, H=H0)
│   │   ├── grpo.py          # GRPO (group mean + std norm)
│   │   ├── rloo.py          # RLOO (leave-one-out, self-excluding)
│   │   └── stv.py           # STV (batch-across baseline)
│   │
│   ├── budgets/             # Budget allocation modules
│   │   ├── base.py          # Abstract BaseBudget
│   │   ├── prompt_skip.py   # Skip low-variance prompts
│   │   ├── rollout_alloc.py # Non-uniform rollout allocation
│   │   └── subset_select.py # Top-k rollout selection
│   │
│   ├── metrics/             # Bias / variance measurement
│   │   ├── bias.py          # Bias decomposition (Theorem 1)
│   │   └── variance.py      # Variance decomposition + HL proxy (Theorem 2)
│   │
│   ├── models/
│   │   └── vlm_wrapper.py   # HuggingFace VLM wrapper
│   │
│   └── utils/
│       ├── probe.py         # Probe vectors for scalar projection
│       └── rollout.py       # RolloutBatch construction utilities
│
├── experiments/
│   └── run_experiment.py    # Main experiment runner (creates timestamped dirs)
│
├── datasets/
│   ├── download.py          # HuggingFace dataset downloader
│   └── __init__.py
│
├── configs/
│   └── default.yaml         # Default hyperparameters
│
├── assets/
│   ├── fig1.png             # Framework overview figure
│   └── fig2.png             # VLM-RL diagnosis figure
│
├── requirements.txt
├── setup.py
├── README.md
└── README_KR.md
```

---

## Baselines & Budget Methods

### Baseline Methods

| Name | Description | $A_B$ | $D_B$ | Bias |
|------|-------------|--------|--------|------|
| **REINFORCE** | No baseline | $0$ | $I$ | Zero (reference) |
| **GRPO** | Group mean + std norm | Block-diag | Reward-dependent | Normalization bias possible |
| **RLOO** | Leave-one-out | Block-diag, diag=0 | $I$ | Zero baseline bias |
| **STV** | Batch-across mean | Full off-diagonal | $I$ | Cross-prompt coupling |

### Budget Methods

| Name | Description | $H$ deviation | Fusion Bias |
|------|-------------|----------------|-------------|
| **None** | $H = H_0$ | Zero | Zero |
| **PromptSkip** | Zero out low-variance prompts | Block-level zeroing | Possible |
| **RolloutAlloc** | Non-uniform $N_j$ allocation | Proportional reweighting | Possible |
| **SubsetSelect** | Top-k rollout selection | Sparse reweighting | Largest among three |

---

## Datasets

CUBE experiments use the following VLM benchmarks (available via HuggingFace Hub):

| Dataset | HF Name | Task | Size |
|---------|---------|------|------|
| **MathVista** | `AI4Math/MathVista` | Math reasoning | 6,141 |
| **MMStar** | `lmms-lab/MMStar` | Multimodal reasoning | 1,500 |
| **ChartQA** | `HuggingFaceM4/ChartQA` | Chart understanding | 32,717 |
| **MMBench** | `lmms-lab/MMBench` | Multi-task | 4,377 |
| **ScienceQA** | `HuggingFaceM4/ScienceQA` | Science with images | 21,208 |
| **MMMU-Pro** | `lmms-lab/MMMU_Pro` | College-level reasoning | 3,460 |
| **VQAv2** | `HuggingFaceM4/VQAv2` | Visual QA | 214,354 |
| **OK-VQA** | `Multimodal-Fatima/OK-VQA_train` | Knowledge VQA | 9,009 |

```bash
# Download a dataset
python datasets/download.py --dataset mathvista --split testmini

# List all available datasets
python datasets/download.py --list
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/khkim1729/cube.git
cd cube

# 2. Install dependencies
pip install -r requirements.txt
# or: pip install -e .

# 3. Download a dataset
python datasets/download.py --dataset mathvista --split testmini

# 4. Run bias/variance sweep (dry run, no GPU required)
python experiments/run_experiment.py --dry_run

# 5. Run with custom config
python experiments/run_experiment.py --config configs/default.yaml --model Qwen/Qwen2-VL-7B-Instruct
```

---

## Experiments

Each experiment run creates a timestamped directory:

```
experiments/
  Qwen_Qwen2_VL_7B_Instruct/
    20250101_120000/
      config.yaml
      metrics/
        bias_variance_heatmap.json
        per_step_*.json
      logs/
```

### Experiment Design (from paper)

**Bias Experiments:**
- `Experiment 1`: Measure step-wise total bias across all 4×3 = 12 baseline × budget combinations → heatmap
- `Experiment 2`: Decompose bias into Budget / Baseline / Fusion components for selected combos

**Variance Experiments:**
- `Experiment 1`: Step-wise total variance for all 12 combinations → heatmap
- `Experiment 2`: Variance decomposition (within-prompt vs across-prompt)
- `Experiment 3`: $\|HL\|_F^2$ proxy accuracy — Spearman correlation with actual variance

**Additional Analysis:**
- Bias-variance vs. final task accuracy scatter plots

### Key Hyperparameters

| Symbol | Default | Description |
|--------|---------|-------------|
| $M$ | 256 | Total rollouts per minibatch |
| $B$ | 32 | Prompts per minibatch |
| $S$ | 16 | Minibatch sampling repetitions |
| $K$ | 8 | Rollout resamples per minibatch |
| $T$ | 10 | Logging checkpoints |
| $R$ | 32 | Probe vector count |

---

## Citation

```bibtex
@article{cube2025,
  title  = {CUBE: Coupled Updates and Budget Estimators in Reinforcement Learning},
  author = {Choi, Minseo},
  year   = {2025},
}
```
