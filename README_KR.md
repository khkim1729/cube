# CUBE: Coupled Updates and Budget Estimators in Reinforcement Learning

> **한국어** | **[English README](README.md)**

---

## 목차

- [개요](#개요)
- [핵심 기여](#핵심-기여)
- [프레임워크 구조](#프레임워크-구조)
- [논문 그림](#논문-그림)
- [코드 구조](#코드-구조)
- [베이스라인 및 예산 방법](#베이스라인-및-예산-방법)
- [데이터셋](#데이터셋)
- [빠른 시작](#빠른-시작)
- [실험](#실험)
- [인용](#인용)

---

## 개요

**CUBE** (**C**oupled **U**pdates and **B**udget **E**stimators)는 검증 가능한 보상 기반 강화학습(RLVR)에서 그래디언트 추정기를 분석하기 위한 통합 이론 및 실험 프레임워크입니다. 특히 멀티모달 비전-언어 모델(VLM)에서의 RL 훈련 안정성에 초점을 맞춥니다.

현대 RLVR 파이프라인은 다음 두 요소를 결합합니다:
- **베이스라인 방법** (REINFORCE, GRPO, RLOO, STV) — 분산 감소를 위한 제어 변량
- **예산 모듈** (프롬프트 스킵, 롤아웃 배분, 서브셋 선택) — 연산 자원 제어

CUBE는 전체 추정기를 다음과 같은 **행렬 형태**로 표현합니다:

$$\hat{g} = \Psi H \tilde{a} = \Psi H D_B (I - A_B) r$$

| 기호 | 의미 |
|------|------|
| $\Psi \in \mathbb{R}^{d \times M}$ | 롤아웃별 스코어(로그 확률 그래디언트) 행렬 |
| $H \in \mathbb{R}^{M \times M}$ | 대각 예산 행렬 |
| $A_B \in \mathbb{R}^{M \times M}$ | 베이스라인 연산자 (프롬프트 간 결합 가능) |
| $D_B \in \mathbb{R}^{M \times M}$ | 정규화 연산자 |
| $r \in \mathbb{R}^M$ | 연결된 보상 벡터 |

---

## 핵심 기여

### 1. 정확한 전역 편향 분해 (정리 1)

$$\mathbb{E}[\hat{g}] - \nabla_\theta J(\theta) = \underbrace{\mathbb{E}[\Psi(H-H_0)r]}_{\text{예산 편향}} - \underbrace{\mathbb{E}[\Psi H_0 A_B r]}_{\text{베이스라인 편향}} - \underbrace{\mathbb{E}[\Psi(H-H_0)A_B r]}_{\text{융합 편향}}$$

**융합 편향(Fusion Bias)**은 예산과 베이스라인이 **동시에** 활성화될 때만 나타나는 교차 항입니다. 각 모듈을 개별적으로 절제(ablation)해서는 탐지할 수 없습니다.

### 2. 잡음 증폭 경계 (정리 2)

$$\mathrm{tr}\, \mathrm{Cov}(\hat{g} \mid X, G) \leq \|\Sigma_r\|_2 \cdot \|\Psi\|_2^2 \cdot \|HL\|_F^2$$

$\|HL\|_F^2$는 라우팅($L$)과 가중치 집중($H$)을 통한 분산 증폭의 경량 프록시입니다.

### 3. 소스별 분산 전파 (명제 1)

$$\mathrm{tr}\, \mathrm{Cov}(\hat{g} \mid X, G) = \sum_{m=1}^M \sigma_m^2 \|g_m\|_2^2, \quad g_m = \Psi H L e_m$$

각 좌표 잡음 소스 $\sigma_m^2$는 라우팅과 예산 증폭에 의해 가중된 항을 기여합니다.

---

## 논문 그림

### 그림 1: CUBE 프레임워크 개요

<p align="center">
  <img src="assets/fig1.png" width="800" alt="CUBE 프레임워크 개요"/>
</p>

**그림 1.** CUBE 프레임워크 개요. 중앙 파이프라인은 베이스라인과 예산 선택을 포함하는 통합 행렬 형태 몬테카를로 추정기를 보여줍니다. 왼쪽 패널은 전역 편향의 정확한 분해(정리 1)를 나타내며 총 편향을 예산 편향, 베이스라인 편향, 융합 상호작용 항으로 분리합니다. 오른쪽 패널은 미니배치 결합 하에서의 분산 증폭(정리 2, 명제 1)을 보여주며, 비대각 잡음 라우팅과 가중치 집중이 조건부 분산을 증폭시키는 방식을 강조합니다.

---

### 그림 2: VLM-RL 진단 및 완화

<p align="center">
  <img src="assets/fig2.png" width="800" alt="CUBE를 이용한 VLM-RL 진단"/>
</p>

**그림 2.** CUBE 프레임워크를 이용한 VLM에서의 강화학습 불안정성 진단 및 완화. 왼쪽 패널은 표준 VLM-RL 훈련에서 베이스라인, 정규화, 적응적 예산 연산자의 결합 스태킹이 잡음 라우팅과 융합 상호작용을 유발하여 높은 분산과 훈련 발산으로 이어지는 과정을 보여줍니다. 오른쪽 패널은 CUBE 행렬 형태 분석을 적용하여 편향을 분해하고 소스별 분산 전파를 추적함으로써 멀티모달 추론 벤치마크에서 안정적인 수렴을 가능하게 합니다.

---

## 코드 구조

```
cube/
├── cube/
│   ├── estimators/          # 베이스라인 그래디언트 추정기
│   │   ├── base.py          # 추상 BaseEstimator, RolloutBatch
│   │   ├── reinforce.py     # REINFORCE (A_B=0, D_B=I, H=H0)
│   │   ├── grpo.py          # GRPO (그룹 평균 + 표준편차 정규화)
│   │   ├── rloo.py          # RLOO (Leave-One-Out, 자기 제외)
│   │   └── stv.py           # STV (배치 전체 베이스라인)
│   │
│   ├── budgets/             # 예산 할당 모듈
│   │   ├── base.py          # 추상 BaseBudget
│   │   ├── prompt_skip.py   # 저분산 프롬프트 스킵
│   │   ├── rollout_alloc.py # 비균일 롤아웃 배분
│   │   └── subset_select.py # 상위-k 롤아웃 선택
│   │
│   ├── metrics/             # 편향 / 분산 측정
│   │   ├── bias.py          # 편향 분해 (정리 1)
│   │   └── variance.py      # 분산 분해 + HL 프록시 (정리 2)
│   │
│   ├── models/
│   │   └── vlm_wrapper.py   # HuggingFace VLM 래퍼
│   │
│   └── utils/
│       ├── probe.py         # 스칼라 투영을 위한 프로브 벡터
│       └── rollout.py       # RolloutBatch 구성 유틸리티
│
├── experiments/
│   └── run_experiment.py    # 메인 실험 실행기 (타임스탬프 디렉토리 생성)
│
├── datasets/
│   ├── download.py          # HuggingFace 데이터셋 다운로더
│   └── __init__.py
│
├── configs/
│   └── default.yaml         # 기본 하이퍼파라미터
│
├── assets/
│   ├── fig1.png             # 프레임워크 개요 그림
│   └── fig2.png             # VLM-RL 진단 그림
│
├── requirements.txt
├── setup.py
├── README.md
└── README_KR.md
```

---

## 베이스라인 및 예산 방법

### 베이스라인 방법

| 이름 | 설명 | $A_B$ | $D_B$ | 편향 특성 |
|------|------|--------|--------|----------|
| **REINFORCE** | 베이스라인 없음 | $0$ | $I$ | 참조 추정기 (편향 없음) |
| **GRPO** | 그룹 평균 + 표준편차 정규화 | 블록 대각 | 보상 의존적 | 정규화 편향 가능 |
| **RLOO** | Leave-One-Out | 블록 대각, 대각=0 | $I$ | 베이스라인 편향 = 0 |
| **STV** | 배치 전체 평균 | 완전 비대각 | $I$ | 프롬프트 간 결합 |

### 예산 방법

| 이름 | 설명 | $H$ 편차 | 융합 편향 |
|------|------|----------|----------|
| **없음** | $H = H_0$ | 없음 | 없음 |
| **PromptSkip** | 저분산 프롬프트 제로화 | 블록 수준 제로화 | 발생 가능 |
| **RolloutAlloc** | 비균일 $N_j$ 배분 | 비례 재가중치 | 발생 가능 |
| **SubsetSelect** | 상위-k 롤아웃 선택 | 희소 재가중치 | 세 방법 중 최대 |

---

## 데이터셋

CUBE 실험은 HuggingFace Hub를 통해 이용 가능한 다음 VLM 벤치마크를 사용합니다:

| 데이터셋 | HF 이름 | 태스크 | 크기 |
|---------|---------|--------|------|
| **MathVista** | `AI4Math/MathVista` | 수학적 추론 | 6,141 |
| **MMStar** | `lmms-lab/MMStar` | 멀티모달 추론 | 1,500 |
| **ChartQA** | `HuggingFaceM4/ChartQA` | 차트 이해 | 32,717 |
| **MMBench** | `lmms-lab/MMBench` | 멀티태스크 | 4,377 |
| **ScienceQA** | `HuggingFaceM4/ScienceQA` | 과학 이미지 QA | 21,208 |
| **MMMU-Pro** | `lmms-lab/MMMU_Pro` | 대학 수준 추론 | 3,460 |
| **VQAv2** | `HuggingFaceM4/VQAv2` | 시각적 질의응답 | 214,354 |
| **OK-VQA** | `Multimodal-Fatima/OK-VQA_train` | 지식 기반 VQA | 9,009 |

```bash
# 데이터셋 다운로드
python datasets/download.py --dataset mathvista --split testmini

# 이용 가능한 데이터셋 목록 확인
python datasets/download.py --list
```

---

## 빠른 시작

```bash
# 1. 저장소 클론
git clone https://github.com/khkim1729/cube.git
cd cube

# 2. 의존성 설치
pip install -r requirements.txt
# 또는: pip install -e .

# 3. 데이터셋 다운로드
python datasets/download.py --dataset mathvista --split testmini
```

---

## 실험 실행 방법

CUBE는 세 가지 레벨의 실험을 지원합니다.

### 단일 실험 실행 (`run_pilot.py`)

한 가지 `(baseline × budget)` 조합을 특정 GPU에서 실행합니다.
결과는 `experiments/results/<run_id>.csv`에 저장됩니다.

```bash
# 형식
python experiments/run_pilot.py \
  --baseline <reinforce|grpo|rloo|stv> \
  --budget   <none|prompt_skip|rollout_alloc|subset_select> \
  --gpu_id   <0|1|2|3> \
  --output_dir experiments/results \
  --M 256 --B 32 --N 8 \
  --S 16 --K 8 --T 10 --R 32 \
  --num_train_steps 200

# 예시 1: RLOO × No-budget, GPU 1
CUDA_VISIBLE_DEVICES=1 python experiments/run_pilot.py \
  --baseline rloo --budget none --gpu_id 0 \
  --output_dir experiments/results

# 예시 2: GRPO × SubsetSelect, GPU 2
CUDA_VISIBLE_DEVICES=2 python experiments/run_pilot.py \
  --baseline grpo --budget subset_select --gpu_id 0 \
  --output_dir experiments/results
```

### 3개 파일럿 실험 병렬 실행 (`launch_pilots.py`)

GPU 1, 2, 3에서 3개 대표 조합을 **동시에** 실행합니다.
파일럿 완료 후 나머지 9개 조합을 자동으로 큐잉합니다.

```bash
# 파일럿 3개만 실행
python experiments/launch_pilots.py \
  --output_dir experiments/results \
  --S 4 --K 2 --T 10 \
  --pilots_only

# 전체 12개 조합 자동 체이닝 실행
python experiments/launch_pilots.py \
  --output_dir experiments/results \
  --S 16 --K 8 --T 10
```

**파일럿 구성:**
| GPU | 조합 | 분석 목적 |
|-----|------|----------|
| GPU 1 | RLOO × None | 이론적 기준선 (편향 최소) |
| GPU 2 | GRPO × SubsetSelect | 융합 편향 + 분산 증폭 확인 |
| GPU 3 | STV × PromptSkip | 크로스-프롬프트 잡음 라우팅 확인 |

### 자동 스케줄러 (`auto_next.py`)

안 돌아간 실험을 자동으로 감지하고 다음 실험을 선택해 실행합니다. 터미널 3개에서 동시에 실행해도 같은 실험이 중복 선택되지 않습니다.

```bash
# 터미널 1 — GPU 1
CUDA_VISIBLE_DEVICES=1 python3 experiments/auto_next.py --gpu_id 1

# 터미널 2 — GPU 2
CUDA_VISIBLE_DEVICES=2 python3 experiments/auto_next.py --gpu_id 2

# 터미널 3 — GPU 3
CUDA_VISIBLE_DEVICES=3 python3 experiments/auto_next.py --gpu_id 3

# 큐 상태만 확인 (실험 실행 없이)
python3 experiments/auto_next.py --gpu_id 1 --status

# 큐 초기화 후 처음부터
python3 experiments/auto_next.py --gpu_id 1 --reset
```

**동작 방식:**
1. `experiments/results/*.csv`를 스캔해 완료된 실험 파악 (rows ≥ T)
2. `fcntl.LOCK_EX`로 `queue.lock` 파일을 잠가 동시 중복 선택 방지
3. `running` 상태인데 PID가 죽어있으면 자동으로 `pending`으로 복원
4. 각 GPU 워커는 큐가 소진될 때까지 실험을 순서대로 자동 실행

**빠른 파일럿 실행** (S, K를 줄여 빠른 검증):
```bash
CUDA_VISIBLE_DEVICES=1 python3 experiments/auto_next.py --gpu_id 1 --S 4 --K 2
```

---

### 결과 병합 (`merge_results.py`)

개별 실험 CSV를 하나의 마스터 CSV로 병합합니다.

```bash
python experiments/merge_results.py --results_dir experiments/results
# → experiments/results/combined_results.csv
```

### CSV 컬럼 설명

| 컬럼 | 설명 |
|------|------|
| `run_id` | 실험 고유 ID (타임스탬프 포함) |
| `date`, `time`, `timestamp` | 측정 날짜/시간 (ISO 8601) |
| `gpu_id` | 사용 GPU |
| `baseline` / `budget` | 실험 조합 |
| `step` / `checkpoint_idx` | 학습 스텝 및 체크포인트 인덱스 |
| `total_bias_norm` | 총 편향 크기 $\|\mathbb{E}[\hat{g}] - \nabla J\|$ |
| `budget_bias_proj_mean` | 예산 편향 $\mathbb{E}[\Psi(H-H_0)r]$ |
| `baseline_bias_proj_mean` | 베이스라인 편향 $\mathbb{E}[\Psi H_0 A_B r]$ |
| `fusion_bias_proj_mean` | **융합 편향** $\mathbb{E}[\Psi(H-H_0)A_B r]$ |
| `total_var_mean` | 총 분산 $\text{tr}\,\text{Cov}(\hat{g})$ |
| `within_var_mean` | 프롬프트 내 분산 $\mathbb{E}[\text{Cov}(\hat{g}\mid X)]$ |
| `across_var_mean` | 프롬프트 간 분산 $\text{Cov}(\mathbb{E}[\hat{g}\mid X])$ |
| `HL_proxy_mean` | 분산 프록시 $\|HL\|_F^2$ |
| `reward_mean` | 해당 체크포인트에서의 평균 보상 |
| `elapsed_seconds` | 경과 시간 (초) |

### 시뮬레이션 실험 결과 (전체 12개 조합)

ToyPolicy (64→128→10 MLP, d=9,610 파라미터), GPU 3× NVIDIA A100 80GB, 조합당 약 55초 소요.
`auto_next.py`로 GPU 1,2,3에서 병렬 실행. 마지막 체크포인트(step 180/200) 기준 측정값.

| 베이스라인 | 예산 | Total Bias | Fusion Bias | HL Proxy $\|HL\|_F^2$ |
|----------|------|-----------|------------|----------------------|
| REINFORCE | None | 0.000 | 0.000 | 3.91e-3 |
| REINFORCE | PromptSkip | 6.7e-5 | 0.000 | 2.93e-3 |
| REINFORCE | SubsetSelect | 1.30e-3 | 0.000 | 7.81e-3 |
| GRPO | None | 1.66e-3 | 0.000 | **1.27e+13** |
| GRPO | PromptSkip | 1.69e-3 | 0.000 | **4.14e+12** |
| GRPO | SubsetSelect | **4.24e-3** | **7.0e-5** | **2.54e+13** |
| RLOO | None | 3.15e-4 | 0.000 | 4.46e-3 |
| RLOO | PromptSkip | 3.14e-4 | 4.0e-6 | 3.35e-3 |
| RLOO | SubsetSelect | 1.12e-3 | 7.1e-5 | 8.93e-3 |
| STV | None | 2.67e-4 | 0.000 | 3.89e-3 |
| STV | PromptSkip | 2.71e-4 | 1.7e-5 | 2.92e-3 |
| STV | SubsetSelect | 1.10e-3 | 4.1e-5 | 7.78e-3 |

**주요 관찰:**
- **GRPO** 계열은 HL proxy가 타 조합 대비 10¹²–10¹³배 높음 — 초기 학습 시 보상 표준편차 → 0으로 $D_B \to \infty$ 발산하는 현상
- **RLOO × None**이 가장 낮은 total_bias와 유한한 HL proxy를 달성 — 이론적 최적 조합 확인
- **Fusion Bias**는 예산($H \neq H_0$)과 베이스라인($A_B \neq 0$)이 동시에 활성화될 때만 0이 아님

> **참고:** 위 결과는 단위벡터로 정규화된 probe ($\|v_r\|=1$)로 측정한 값입니다. 현재 코드는 $v_r \sim \mathcal{N}(0,I)$ 에서 정규화 없이 뽑습니다 (`cube/utils/probe.py`). 절대적인 bias 수치는 probe norm (~$\sqrt{d}$)에 비례해 바뀌지만, 조합 간 상대 순서와 $\|HL\|_F^2$는 영향 없음.

### GPU 3 검증 실행 (probe 정규화 제거 후)

probe 정규화 제거 후 **RLOO × None**을 GPU 3에서 재실행 (`S=4, K=2`):

| 베이스라인 | 예산 | Total Bias | Fusion Bias | HL Proxy $\|HL\|_F^2$ |
|----------|------|-----------|------------|----------------------|
| RLOO | None | 3.08e-2 | 0.000 | **4.46e-3** |
| GRPO | SubsetSelect | 4.15e-1 | 6.9e-3 | **2.54e+13** |

**주요 관찰:**
- $\|HL\|_F^2$ 값은 pilot 실험 결과와 **완전히 동일** (RLOO×None: 4.46e-3, GRPO×SubSel: 2.54e+13) — probe 정규화 유무와 무관
- Total Bias는 비정규화 probe 기준 약 ~$\sqrt{d} \approx 98$배 증가 (스케일 변화, 상대 순서 유지)
- RLOO×None의 Fusion Bias는 두 설정 모두 정확히 0 — 이론적 보장 확인

### 실험 설계 (plans_cube_02.txt 기반)

**편향(Bias) 실험:**
- **실험 1**: 4×3 = 12개 (베이스라인 × 예산) 조합에서 학습 중 스텝별 전체 편향 측정 → 히트맵
- **실험 2**: 선택된 조합에 대해 편향을 예산/베이스라인/융합 구성요소로 분해

**분산(Variance) 실험:**
- **실험 1**: 12개 조합에서 스텝별 전체 분산 측정 → 히트맵
- **실험 2**: 분산 분해 (프롬프트 내부 vs 프롬프트 간)
- **실험 3**: $\|HL\|_F^2$ 프록시 정확도 — 실제 분산과의 Spearman 상관계수

### 주요 하이퍼파라미터 (plans_cube_02.txt)

| 기호 | 기본값 | 설명 |
|------|--------|------|
| $M$ | 256 | 미니배치당 전체 롤아웃 수 ($= B \times N_j$) |
| $B$ | 32 | 미니배치당 프롬프트 수 |
| $N_j$ | 8 | 프롬프트당 롤아웃 수 (균일 배분 기준) |
| $S$ | 16 | 미니배치 샘플링 반복 횟수 (Monte-Carlo) |
| $K$ | 8 | 고정 미니배치에서 롤아웃 재샘플 횟수 |
| $T$ | 10 | 학습 중 로깅 체크포인트 수 |
| $R$ | 32 | probe 벡터 수 (시드 42로 고정 생성) |

---

## 인용

```bibtex
@article{cube2025,
  title  = {CUBE: Coupled Updates and Budget Estimators in Reinforcement Learning},
  author = {Choi, Minseo},
  year   = {2025},
}
```
