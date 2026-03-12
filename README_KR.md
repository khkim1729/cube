# SELSA-FiLM: 조영 초음파 간 병변 검출을 위한 위상 조건부 비디오 객체 검출

[English README](README.md)

---

## 목차

1. [소개](#1-소개)
2. [논문 개요](#2-논문-개요)
   - 2.1 [문제 정의](#21-문제-정의)
   - 2.2 [RPN Recall 분석](#22-rpn-recall-분석)
   - 2.3 [SELSA-FiLM 제안](#23-selsa-film-제안)
   - 2.4 [주요 기여](#24-주요-기여)
3. [아키텍처 상세](#3-아키텍처-상세)
   - 3.1 [전체 파이프라인](#31-전체-파이프라인)
   - 3.2 [FiLM 적용 위치](#32-film-적용-위치)
   - 3.3 [SelsaRoIHeadFiLM](#33-selsaroiheadfilm)
   - 3.4 [FasterRCNNFiLM (Backbone/Neck 조건화)](#34-fasterrcnnfilm-backboneneck-조건화)
4. [코드 구조](#4-코드-구조)
5. [환경 설정](#5-환경-설정)
6. [데이터 준비](#6-데이터-준비)
7. [학습 및 평가](#7-학습-및-평가)
   - 7.1 [Config 구조](#71-config-구조)
   - 7.2 [학습](#72-학습)
   - 7.3 [평가 (Test)](#73-평가-test)
   - 7.4 [전체 실험 스크립트](#74-전체-실험-스크립트)
8. [평가 지표](#8-평가-지표)
9. [실험 결과 및 분석](#9-실험-결과-및-분석)
10. [그림 설명](#10-그림-설명)

---

## 1. 소개

**SELSA-FiLM**은 조영증강 초음파(CEUS: Contrast-Enhanced UltraSound)에서 간 병변 검출 및 분류를 위해 설계된 위상 조건부(phase-conditioned) 비디오 객체 검출 방법입니다.

이 저장소는 MMTracking 기반으로 구현되었으며, **`vanilla_FiLM` 브랜치**에서 SELSA-FiLM 관련 코드를 확인할 수 있습니다.

> **논문명:**
> *SELSA-FiLM: Phase-Conditioned Video Object Detection for Liver Lesion Detection and Classification in Contrast-Enhanced Ultrasound*

<img src="resources/figs/fig1.png" alt="Figure 1: SELSA-FiLM 개요" width="100%"/>

*Figure 1. CEUS의 위상 구조적 시계열 문제와 제안 방법 SELSA-FiLM 개요. 자연 비디오와 달리, CEUS 프레임은 동맥기(arterial), 문맥/후기(portal/late), 쿠퍼기(Kupffer) 등 이산적인 생리학적 상태를 인코딩한다. 표준 비디오 객체 검출 방법은 시간적 집합(temporal aggregation) 과정에서 위상 판별 신호를 희석시킨다. SELSA-FiLM은 위상 레이블을 사전 조건으로 활용하여 위상별 특징 분포를 유지한다.*

---

## 2. 논문 개요

### 2.1 문제 정의

비디오 객체 검출(VID) 방법은 프레임 간 객체가 일관되게 나타난다는 가정(temporal consistency assumption)에 기반합니다. 그러나 **CEUS에서는 이 가정이 구조적으로 위반됩니다.**

CEUS에서는 마이크로버블 조영제가 간 혈관계를 통해 이동함에 따라 동일한 병변이 세 가지 위상에서 근본적으로 다른 외관을 보입니다:

| 위상 | 특징 |
|------|------|
| 동맥기 (Arterial) | 과증강 (hyperenhancement) |
| 문맥/후기 (Portal/Late) | 점진적 세척 (progressive washout) |
| 쿠퍼기 (Kupffer) | 거의 완전한 어둠 (near-complete darkness) |

HCC(간세포암)와 FNH(국소 결절성 과형성)는 각 위상에서 진단적으로 구별되는 증강 패턴을 보입니다.

**핵심 성능 격차 발견:**
- SELSA 기준 단순 검출(C1): **53.1 mAP@50**
- SELSA 기준 검출+분류(C2): **33.2 mAP@50**
- **약 20 mAP 포인트 차이** — 모든 모델 변형과 병변 카테고리에서 일관되게 나타남

### 2.2 RPN Recall 분석

성능 격차의 근본 원인을 파악하기 위해 **위상별 RPN Recall 분해 분석**을 수행합니다.

```
위상별 Region Proposal Recall@50 (C1 병변 클래스):
  - 동맥기 (Arterial):    71.0%
  - 문맥/후기 (Portal/Late): 39.8%
  - 쿠퍼기 (Kupffer):     49.9% ± 17.2%
```

**진단 결과:** 제안(proposal) 생성 단계 자체가 위상에 민감하며, 병변 조영이 약해질수록 적절한 후보를 생성하지 못합니다. 이는 위상 정보가 특징 추출 단계(backbone)에서 네트워크에 진입해야 함을 직접적으로 시사합니다.

### 2.3 SELSA-FiLM 제안

**FiLM(Feature-wise Linear Modulation)**을 SELSA의 세 가지 아키텍처 위치에 통합합니다:

```
입력 → [Backbone] → [Neck] → RPN → [RoI Head] → 출력
         ↑ FiLM        ↑ FiLM           ↑ FiLM
         (위상 조건화)
```

FiLM은 중간 특징 맵에 위상별 아핀 변환(affine transformation)을 학습합니다:

$$\tilde{F} = \gamma(\text{phase}) \odot F + \beta(\text{phase})$$

여기서 $\gamma$와 $\beta$는 위상 임베딩(phase embedding)을 통해 학습됩니다. CEUS 획득 프로토콜은 위상 레이블을 **사전에(a priori)** 제공하여 깔끔하고 이산적이며 노이즈 없는 조건화 신호를 공급합니다.

<img src="resources/figs/fig2.png" alt="Figure 2: SELSA-FiLM 상세 아키텍처" width="100%"/>

*Figure 2. SELSA-FiLM의 세 위치 위상 조건화 상세 아키텍처. FiLM은 백본(backbone), 특징 피라미드 넥(feature pyramid neck), RoI 헤드(RoI head)의 세 위치에 통합된다. 네트워크는 사전 위상 레이블을 활용하여 위상별 아핀 변환 파라미터(γ, β)를 학습한다.*

### 2.4 주요 기여

1. **문제 정의**: CEUS와 같이 VID의 시간적 일관성 가정이 구조적으로 위반되는 "위상 구조적 시계열(phase-structured temporal sequence)" 개념 제시. SELSA에서 검출-분류 간 ~20 mAP 격차 실증.

2. **진단 분석**: 위상별 RPN Recall 분해를 통한 VID 방법의 컴포넌트 수준 실패 분석 최초 제시. 동맥기 71.0% → 문맥/후기 39.8%로 recall 급락.

3. **방법론**: 세 아키텍처 위치에 FiLM 조건화를 통합한 SELSA-FiLM 제안. 체계적 ablation을 통해 백본 수준 위상 조건화가 가장 효과적임을 입증.

4. **평가 프로토콜**: 4개 병변 카테고리, 5-fold 교차검증으로 평가. 검출(C1)과 검출+분류(C2) 성능을 분리하여 정밀한 진단 가능.

---

## 3. 아키텍처 상세

### 3.1 전체 파이프라인

```
CEUS 비디오 시퀀스
│
├── 키 프레임 (key frame, 위상: arterial/portal/kupffer)
└── 레퍼런스 프레임들 (reference frames, 각각 위상 정보 포함)
           │
           ▼
    [FasterRCNNFiLM / SELSA]
           │
    ┌──────┴──────┐
    │   Backbone  │ ← FiLM 적용 (backbone 모드)
    │  (ResNet50) │   위상 임베딩 → γ, β
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │    Neck     │ ← FiLM 적용 (neck 모드)
    │(ChannelMapper│   위상 임베딩 → γ, β
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  RPN Head   │  위상별 Region Proposal 생성
    └──────┬──────┘
           │
    ┌──────┴──────────────────┐
    │  SELSA Aggregation      │  자기주의(self-attention) 기반
    │  (키↔레퍼런스 어텐션)   │  시간적 특징 집합
    └──────┬──────────────────┘
           │
    ┌──────┴──────┐
    │  RoI Head   │ ← FiLM 적용 (roi 모드, SelsaRoIHeadFiLM)
    │             │   위상 임베딩 → γ, β
    └──────┬──────┘
           │
    검출 결과 (bbox + class)
```

### 3.2 FiLM 적용 위치

세 가지 독립적인 FiLM 적용 모드를 지원합니다:

| 모드 | 적용 위치 | 조건화 채널 수 | 구현 위치 |
|------|-----------|----------------|-----------|
| `backbone` | Backbone 출력 | 2048 | `FasterRCNNFiLM` (mmdet) |
| `neck` | Neck 출력 | 512 | `FasterRCNNFiLM` (mmdet) |
| `roi` | RoI 특징 | 512 | `SelsaRoIHeadFiLM` (mmtrack) |

**Config 별 설정 예시:**

```python
# Backbone FiLM (film_backbone)
film_cfg=dict(
    apply_to='backbone',
    num_phases=3,
    emb_dim=32,
    channels=2048,
    baseline=True,  # gamma = 1 + gamma (항등 변환에서 초기화)
    debug=True
)

# Neck FiLM (film_neck)
film_cfg=dict(
    apply_to='neck',
    num_phases=3,
    emb_dim=32,
    channels=512,
    baseline=True,
    debug=True
)

# RoI FiLM (film_roi) - SelsaRoIHeadFiLM 사용
roi_head=dict(
    type='mmtrack.SelsaRoIHeadFiLM',
    film_cfg=dict(
        num_phases=3,
        emb_dim=32,
        channels=512,  # RoIAlign 출력 채널
        baseline=True,
        debug=True
    )
)
```

### 3.3 SelsaRoIHeadFiLM

**파일:** `mmtrack/models/roi_heads/selsa_roi_head_film.py`

`SelsaRoIHead`를 상속하여 ROIAlign 출력 직후에 FiLM을 적용합니다.

```python
@MODELS.register_module()
class SelsaRoIHeadFiLM(SelsaRoIHead):
    """
    RoI 특징에 FiLM을 적용하는 SELSA RoI 헤드.
    - Key 프레임: 배치 전체에 동일 위상으로 FiLM 적용
    - Ref 프레임: 프레임별 ROI 개수로 분할 후, 각 프레임 위상으로 FiLM 적용
    """
```

**핵심 컴포넌트:**

```python
# 위상 임베딩 레이어
self.phase_emb = nn.Embedding(num_phases, emb_dim)  # (3, 32)

# FiLM MLP: 임베딩 → gamma, beta
self.film_mlp = nn.Linear(emb_dim, 2 * film_channels)  # (32, 1024)

# 초기화: 항등 변환 (gamma=1, beta=0)
nn.init.zeros_(self.film_mlp.weight)
nn.init.zeros_(self.film_mlp.bias)
```

**FiLM 파라미터 계산:**

```python
def _film_params(self, phase_id):
    z = self.phase_emb(phase_id)   # (B, emb_dim)
    gb = self.film_mlp(z)          # (B, 2*C)
    gamma, beta = gb.chunk(2, dim=1)
    if self.use_baseline:
        gamma = 1.0 + gamma        # 항등 변환에서 시작
    return gamma.view(-1, C, 1, 1), beta.view(-1, C, 1, 1)
```

**레퍼런스 프레임 처리:**

레퍼런스 프레임은 여러 위상에서 올 수 있으므로, 프레임별 proposal 개수에 따라 분할 후 각각의 위상으로 FiLM을 적용합니다:

```python
def _apply_film_split_by_list(self, feats, per_img_counts, phase_ids):
    """
    feats: (sum_K, C, H, W)  - 모든 ref frame의 ROI 특징 합산
    per_img_counts: (K1, K2, ...)  - 각 ref frame의 proposal 수
    phase_ids: (num_imgs,)  - 각 ref frame의 위상 ID
    """
    chunks = feats.split(per_img_counts, dim=0)
    return torch.cat([
        self._apply_film_batchwise(ch, phase_ids[i:i+1])
        for i, ch in enumerate(chunks)
    ])
```

### 3.4 FasterRCNNFiLM (Backbone/Neck 조건화)

**파일:** `configs/_base_/models/faster-rcnn_r50-dc5-FiLM.py`

Backbone 또는 Neck에 FiLM을 적용하는 경우, MMDetection의 `FasterRCNNFiLM` 모듈을 사용합니다.

```python
model = dict(
    detector=dict(
        type='FasterRCNNFiLM',
        _scope_='mmdet',
        backbone=dict(
            type='ResNet',
            depth=50,
            out_indices=(3,),          # stage4 출력 (2048 채널)
            strides=(1, 2, 2, 1),
            dilations=(1, 1, 1, 2),    # DC5 설정
        ),
        neck=dict(
            type='ChannelMapper',
            in_channels=[2048],
            out_channels=512,          # 넥 출력 512 채널
        ),
        ...
    )
)
```

---

## 4. 코드 구조

```
CEUS_mmtracking/
├── mmtrack/
│   └── models/
│       ├── vid/
│       │   └── selsa.py                    # SELSA 비디오 검출기 (FiLM 지원 수정)
│       ├── roi_heads/
│       │   ├── selsa_roi_head.py           # 기본 SELSA RoI 헤드
│       │   └── selsa_roi_head_film.py      # ★ FiLM 적용 RoI 헤드 (신규)
│       └── aggregators/
│           └── selsa_aggregator.py         # SELSA 자기주의 집합기
│
├── configs/
│   ├── _base_/
│   │   └── models/
│   │       └── faster-rcnn_r50-dc5-FiLM.py  # ★ FiLM 기본 모델 config (신규)
│   │
│   └── vid/
│       ├── selsa_ceus_film_backbone/       # ★ Backbone FiLM 실험 (신규)
│       │   ├── film_backbone-12e_c1_fold{0-4}.py
│       │   ├── film_backbone-12e_c2_fold{0-4}.py
│       │   ├── film_backbone-12e_fnh_fold{0-4}.py
│       │   ├── film_backbone-12e_hcc_fold{0-4}.py
│       │   ├── train.sh
│       │   └── train_and_test.sh
│       │
│       ├── selsa_ceus_film_neck/           # ★ Neck FiLM 실험 (신규)
│       │   ├── film_neck-12e_{dataset}_fold{0-4}.py  (×20)
│       │   ├── train.sh
│       │   └── train_and_test.sh
│       │
│       └── selsa_ceus_film_roi/            # ★ RoI FiLM 실험 (신규)
│           ├── film_roi-12e_{dataset}_fold{0-4}.py  (×20)
│           ├── train.sh
│           └── train_and_test.sh
│
└── resources/
    └── figs/
        ├── fig1.png                        # 방법론 개요 그림
        └── fig2.png                        # 상세 아키텍처 그림
```

**vanilla_FiLM 브랜치에서 추가된 핵심 파일:**

| 파일 | 설명 |
|------|------|
| `mmtrack/models/roi_heads/selsa_roi_head_film.py` | RoI 수준 FiLM 적용 헤드 |
| `configs/_base_/models/faster-rcnn_r50-dc5-FiLM.py` | FiLM 기본 모델 설정 |
| `configs/vid/selsa_ceus_film_backbone/` | Backbone FiLM 실험 설정 (20개) |
| `configs/vid/selsa_ceus_film_neck/` | Neck FiLM 실험 설정 (20개) |
| `configs/vid/selsa_ceus_film_roi/` | RoI FiLM 실험 설정 (20개) |

---

## 5. 환경 설정

```bash
# 1. conda 환경 생성
conda create -n mmlab python=3.9 -y
conda activate mmlab

# 2. PyTorch 설치
conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch

# 3. mmengine 설치
pip install 'mmengine==0.10.7'

# 4. mmcv 설치
pip install 'mmcv==2.0.0rc4' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

# 5. mmdetection 설치
git clone -b v3.0.0rc5 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
conda install "numpy<2"
pip install -v -e . --no-build-isolation
cd ..

# 6. mmtracking (CEUS_mmtracking) 설치
git clone https://github.com/khkim1729/cube.git CEUS_mmtracking
cd CEUS_mmtracking
git checkout vanilla_FiLM
pip install -r requirements/build.txt
pip install -v -e . --no-build-isolation

# 7. 추가 라이브러리
pip install git+https://github.com/JonathonLuiten/TrackEval.git
pip uninstall -y opencv-python
pip install 'opencv-python==4.7.0.72'
```

---

## 6. 데이터 준비

### 디렉토리 구조

```
CEUS_mmtracking/
└── data/
    └── CEUS/
        ├── annotations/
        │   └── video/
        │       ├── c1/
        │       │   ├── ceus_video_c1_fold0_train.json
        │       │   ├── ceus_video_c1_fold0_val.json
        │       │   ├── ceus_video_c1_fold0_test.json
        │       │   └── ... (fold1 ~ fold4)
        │       ├── c2/
        │       ├── fnh/
        │       └── hcc/
        │
        ├── Annotations/       # 원본 JSON (패딩 포함)
        │   ├── post_padding.json
        │   ├── pre_padding.json
        │   └── rand_padding.json
        │
        └── Data/              # 원본 CEUS 영상
            ├── fold_0/
            ├── fold_1/
            ├── fold_2/
            ├── fold_3/
            └── fold_4/
```

### COCOVID 형식 변환

```bash
# 1. COCO 형식으로 변환
python tools/dataset_converters/ceus/ceus2coco.py

# 2. 빈 프레임 처리
python tools/dataset_converters/ceus/fill_blank.py
```

### 데이터셋 카테고리 정의

| 카테고리 | 설명 |
|----------|------|
| **C1** | 단일 클래스 검출 (병변 여부만 판단) |
| **C2** | 다중 클래스 검출+분류 (HCC vs FNH vs 기타) |
| **FNH** | 국소 결절성 과형성 단독 |
| **HCC** | 간세포암 단독 |

---

## 7. 학습 및 평가

### 7.1 Config 구조

Config 이름 형식: `{film_적용위치}-{epoch}e_{데이터셋}_fold{k}.py`

```
film_backbone-12e_c1_fold0.py
  │           │   │      │
  │           │   │      └── 5-fold 교차검증 인덱스 (0~4)
  │           │   └───────── 데이터셋 (c1, c2, fnh, hcc)
  │           └───────────── 학습 에폭 수
  └───────────────────────── FiLM 적용 위치 (backbone, neck, roi)
```

### 7.2 학습

```bash
# 단일 실험 학습
python tools/train.py \
    configs/vid/selsa_ceus_film_backbone/film_backbone-12e_c1_fold0.py \
    --work-dir experiments/film_backbone_c1_fold0

# 특정 위치의 모든 실험 (학습만)
cd configs/vid/selsa_ceus_film_backbone
bash train.sh
```

### 7.3 평가 (Test)

```bash
# 단일 체크포인트 평가
python tools/test.py \
    configs/vid/selsa_ceus_film_backbone/film_backbone-12e_c1_fold0.py \
    --checkpoint experiments/film_backbone_c1_fold0/best_coco_bbox_mAP_50_epoch_X.pth \
    --work-dir experiments/film_backbone_c1_fold0
```

평가 지표로 **CocoVideoMetric**과 **PhaseRPNRecall**이 동시에 계산됩니다:

```python
test_evaluator = [
    dict(type='CocoVideoMetric',    # mAP@50 등 COCO 메트릭
         ann_file=data_root + test_file,
         metric='bbox'),
    dict(type='PhaseRPNRecall',     # 위상별 RPN Recall 분석
         Ks=(50, 100, 300),
         iou_thr=0.5)
]
```

### 7.4 전체 실험 스크립트

모든 fold × 데이터셋 조합을 순차적으로 학습+평가합니다:

```bash
# FiLM 적용 위치별 전체 실험
bash configs/vid/selsa_ceus_film_backbone/train_and_test.sh  # Backbone
bash configs/vid/selsa_ceus_film_neck/train_and_test.sh      # Neck
bash configs/vid/selsa_ceus_film_roi/train_and_test.sh       # RoI
```

스크립트 내부에서 `c1/c2/fnh/hcc` × `fold0~4` = **20개 실험**을 자동으로 수행합니다.

---

## 8. 평가 지표

### mAP@50 (COCO Video Metric)

표준 COCO 형식의 Bounding Box mAP를 영상 단위로 계산합니다.

```
C1: 단일 클래스 검출 성능 (병변 위치 검출 능력)
C2: 다중 클래스 검출+분류 성능 (검출 + 병변 타입 분류)
```

### PhaseRPNRecall

위상별 RPN Recall을 분해하여 Proposal 생성 단계의 위상 민감도를 정량화합니다:

```python
dict(type='PhaseRPNRecall',
     Ks=(50, 100, 300),   # Top-K proposals
     iou_thr=0.5)         # IoU threshold
```

출력 예시:
```
Phase RPN Recall@50:
  Arterial:    71.0%
  Portal/Late: 39.8%
  Kupffer:     49.9% ± 17.2%
```

---

## 9. 실험 결과 및 분석

### FiLM 위치별 Ablation

5-fold 교차검증 평균 성능 (C1 mAP@50):

| 방법 | C1 (검출) | C2 (검출+분류) | ΔC2 향상 |
|------|-----------|----------------|----------|
| SELSA (Baseline) | 53.1 | 33.2 | - |
| SELSA-FiLM (Backbone) | - | - | **최우수** |
| SELSA-FiLM (Neck) | - | - | - |
| SELSA-FiLM (RoI) | - | - | - |

> 구체적인 수치는 논문 본문을 참조하세요. Backbone 수준 FiLM이 가장 일관된 성능 향상을 보였으며, 이는 RPN Recall 분석에서 진단한 상류(upstream) 실패를 직접적으로 교정합니다.

### 핵심 발견

1. **Backbone FiLM이 가장 효과적**: RPN Recall 분석에서 상류 단계가 병목임을 진단했고, Backbone에 FiLM을 적용하면 이를 직접 교정.
2. **위상 정보는 a priori 활용 가능**: CEUS 프로토콜이 위상 레이블을 제공하므로 추가 학습 없이 조건화 가능.
3. **FiLM의 안정적 초기화**: `gamma = 1 + gamma` (baseline=True)로 초기화하여 학습 초기 안정성 확보.

---

## 10. 그림 설명

### Figure 1

![Fig1](resources/figs/fig1.png)

**개요 그림**: CEUS에서 위상 구조적 시계열의 문제와 SELSA-FiLM 프레임워크 개요.
자연 비디오와 달리, CEUS 프레임은 이산적인 생리학적 상태(동맥기, 문맥/후기, 쿠퍼기)를 인코딩합니다. 표준 비디오 객체 검출 방법은 시간적 집합 과정에서 위상 판별 신호를 희석시킵니다. SELSA-FiLM은 이미징 프로토콜의 사전 위상 레이블을 활용하여 네트워크를 조건화하고, 위상별 특징을 보존합니다.

### Figure 2

![Fig2](resources/figs/fig2.png)

**상세 아키텍처 그림**: SELSA-FiLM의 세 위치 위상 조건화 상세 구조.
병변 조영 감소로 인한 상류 Region Proposal 실패를 해결하기 위해, FiLM이 백본, 특징 피라미드 넥, RoI 헤드 세 위치에 통합됩니다. 네트워크는 사전 위상 레이블을 이용하여 위상별 아핀 변환 파라미터(γ, β)를 학습하고, 이를 통해 자기주의 기반 시간적 집합 이전에 각 생리학적 상태에 대한 구별되는 특징 분포를 유지합니다.

---

## 라이선스

이 프로젝트는 Apache 2.0 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.

---

*이 문서는 `vanilla_FiLM` 브랜치 기준으로 작성되었습니다.*
