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

# 4. 편향/분산 스윕 실행 (dry run, GPU 불필요)
python experiments/run_experiment.py --dry_run

# 5. 사용자 정의 설정으로 실행
python experiments/run_experiment.py --config configs/default.yaml --model Qwen/Qwen2-VL-7B-Instruct
```

---

## 실험

각 실험 실행은 타임스탬프가 포함된 디렉토리를 생성합니다:

```
experiments/
  Qwen_Qwen2_VL_7B_Instruct/
    20250101_120000/
      config.yaml
      metrics/
        bias_variance_heatmap.json
      logs/
```

### 실험 설계

**편향(Bias) 실험:**
- **실험 1**: 4×3 = 12개 (베이스라인 × 예산) 조합에서 학습 중 스텝별 전체 편향 측정 → 히트맵
- **실험 2**: 선택된 조합에 대해 편향을 예산/베이스라인/융합 구성요소로 분해

**분산(Variance) 실험:**
- **실험 1**: 12개 조합에서 스텝별 전체 분산 측정 → 히트맵
- **실험 2**: 분산 분해 (프롬프트 내부 vs 프롬프트 간)
- **실험 3**: $\|HL\|_F^2$ 프록시 정확도 — 실제 분산과의 Spearman 상관계수

**추가 분석:**
- 편향-분산 vs 최종 태스크 정확도 히트맵

### 주요 하이퍼파라미터

| 기호 | 기본값 | 설명 |
|------|--------|------|
| $M$ | 256 | 미니배치당 전체 롤아웃 수 |
| $B$ | 32 | 미니배치당 프롬프트 수 |
| $S$ | 16 | 미니배치 샘플링 반복 횟수 |
| $K$ | 8 | 고정 미니배치에서 롤아웃 재샘플 횟수 |
| $T$ | 10 | 로깅 체크포인트 수 |
| $R$ | 32 | 프로브 벡터 수 |

---

## 인용

```bibtex
@article{cube2025,
  title  = {CUBE: Coupled Updates and Budget Estimators in Reinforcement Learning},
  author = {Choi, Minseo},
  year   = {2025},
}
```
