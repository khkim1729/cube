# CEUS_mmtracking

**[한국어 문서 README_KR.md](README_KR.md)**

---

## Table of Contents

1. [Introduction](#introduction)
2. [SELSA-FiLM (vanilla_FiLM Branch)](#selsa-film-vanilla_film-branch)
   - [Overview](#overview)
   - [Architecture](#architecture)
   - [FiLM Injection Sites](#film-injection-sites)
   - [New Files Added](#new-files-added)
3. [Benchmark and Model Zoo](#benchmark-and-model-zoo)
4. [Setup for CEUS](#setup-for-ceus)
5. [Data Preparation](#data-preparation)
6. [Training & Testing](#training--testing)

---

## Introduction

MMTracking is an open source video perception toolbox by PyTorch. It is a part of [OpenMMLab](https://openmmlab.com) project.

The master branch works with **PyTorch1.6+**.

Please refer to [get_started.md](docs/en/get_started.md) for install instructions.

---

## SELSA-FiLM (vanilla_FiLM Branch)

### Overview

> **Paper:** *SELSA-FiLM: Phase-Conditioned Video Object Detection for Liver Lesion Detection and Classification in Contrast-Enhanced Ultrasound*

The `vanilla_FiLM` branch extends the SELSA video object detector with **Feature-wise Linear Modulation (FiLM)** to address a fundamental challenge in CEUS: the **phase-structured temporal sequence** problem.

Unlike natural videos where temporal variation reflects object motion, CEUS frames encode discrete physiological states (arterial, portal/late, and Kupffer phases). This causes VID methods to suffer a ~20 mAP performance gap between detection-only (C1: 53.1 mAP@50) and joint detection-and-classification (C2: 33.2 mAP@50) tasks.

**Root Cause (RPN Recall Analysis):**
| Phase | RPN Recall@50 |
|-------|---------------|
| Arterial | 71.0% |
| Portal/Late | 39.8% |
| Kupffer | 49.9% ± 17.2% |

Proposal generation itself is phase-sensitive. FiLM conditioning at the backbone level directly addresses this upstream failure.

<img src="resources/figs/fig1.png" alt="Figure 1: SELSA-FiLM Overview" width="100%"/>

*Figure 1. Overview of the phase-structured temporal sequence challenge in CEUS and the proposed SELSA-FiLM framework. SELSA-FiLM leverages a priori phase labels from the imaging protocol to condition the network, preserving phase-specific features.*

### Architecture

SELSA-FiLM integrates FiLM at three architectural sites:

```
Input → [Backbone] → [Neck] → RPN → [RoI Head] → Output
            ↑ FiLM      ↑ FiLM          ↑ FiLM
         (phase conditioning)
```

FiLM applies per-phase affine transformations to intermediate feature maps:

$$\tilde{F} = \gamma(\text{phase}) \odot F + \beta(\text{phase})$$

where γ and β are learned via a phase embedding and a linear layer. Initialization uses `gamma = 1 + gamma` (identity at init).

<img src="resources/figs/fig2.png" alt="Figure 2: SELSA-FiLM Architecture" width="100%"/>

*Figure 2. Detailed architecture of SELSA-FiLM featuring three-site phase conditioning. FiLM is integrated at the backbone, feature pyramid neck, and RoI head.*

### FiLM Injection Sites

| Mode | Location | Channels | Implementation |
|------|----------|----------|----------------|
| `backbone` | After ResNet50 stage4 | 2048 | `FasterRCNNFiLM` |
| `neck` | After ChannelMapper | 512 | `FasterRCNNFiLM` |
| `roi` | After RoIAlign | 512 | `SelsaRoIHeadFiLM` |

Each mode has 20 configs: 4 datasets (c1, c2, fnh, hcc) × 5 folds.

### New Files Added

| File | Description |
|------|-------------|
| `mmtrack/models/roi_heads/selsa_roi_head_film.py` | RoI-level FiLM head (`SelsaRoIHeadFiLM`) |
| `configs/_base_/models/faster-rcnn_r50-dc5-FiLM.py` | Base model config with FiLM support |
| `configs/vid/selsa_ceus_film_backbone/` | Backbone FiLM experiments (20 configs) |
| `configs/vid/selsa_ceus_film_neck/` | Neck FiLM experiments (20 configs) |
| `configs/vid/selsa_ceus_film_roi/` | RoI FiLM experiments (20 configs) |
| `resources/figs/fig1.png` | Method overview figure |
| `resources/figs/fig2.png` | Detailed architecture figure |

---

## Benchmark and Model Zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

### Video Object Detection

Supported Methods

- [x] [DFF](configs/vid/dff) (CVPR 2017)
- [x] [FGFA](configs/vid/fgfa) (ICCV 2017)
- [x] [SELSA](configs/vid/selsa) (ICCV 2019)
- [x] [Temporal RoI Align](configs/vid/temporal_roi_align) (AAAI 2021)
- [x] **SELSA-FiLM** (vanilla_FiLM branch) — CEUS phase-conditioned VID

Supported Datasets

- [x] [ILSVRC](http://image-net.org/challenges/LSVRC/2015/)
- [x] CEUS (internal dataset: C1, C2, FNH, HCC)

### Multi-Object Tracking

Supported Methods

- [x] [SORT](configs/mot/sort) (ICIP 2016)
- [x] [DeepSORT](configs/mot/deepsort) (ICIP 2017)
- [x] [Tracktor](configs/mot/tracktor) (ICCV 2019)
- [x] [QDTrack](configs/mot/qdtrack) (CVPR 2021)
- [x] [ByteTrack](configs/mot/bytetrack) (ECCV 2022)
- [x] [StrongSORT](configs/mot/strongsort) (arxiv 2022)

Supported Datasets

- [x] [MOT Challenge](https://motchallenge.net/)
- [x] [CrowdHuman](https://www.crowdhuman.org/)
- [x] [LVIS](https://www.lvisdataset.org/)
- [x] [TAO](https://taodataset.org/)
- [x] [DanceTrack](https://arxiv.org/abs/2111.14690)

### Video Instance Segmentation

Supported Methods

- [x] [MaskTrack R-CNN](configs/vis/masktrack_rcnn) (ICCV 2019)
- [x] [Mask2Former](configs/vis/mask2former) (CVPR 2022)

Supported Datasets

- [x] [YouTube-VIS](https://youtube-vos.org/dataset/vis/)

### Single Object Tracking

Supported Methods

- [x] [SiameseRPN++](configs/sot/siamese_rpn) (CVPR 2019)
- [x] [PrDiMP](configs/sot/prdimp) (CVPR2020)
- [x] [STARK](configs/sot/stark) (ICCV 2021)

Supported Datasets

- [x] [LaSOT](http://vision.cs.stonybrook.edu/~lasot/)
- [x] [UAV123](https://cemse.kaust.edu.sa/ivul/uav123/)
- [x] [TrackingNet](https://tracking-net.org/)
- [x] [OTB100](http://www.visual-tracking.net/)
- [x] [GOT10k](http://got-10k.aitestunion.com/)
- [x] [VOT2018](https://www.votchallenge.net/vot2018/)

---

## Setup for CEUS

```bash
# 1. Create conda environment
conda create -n mmlab python=3.9 -y
conda activate mmlab

# 2. Install PyTorch
conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch

# 3. Install mmengine
pip install 'mmengine==0.10.7'

# 4. Install mmcv
pip install 'mmcv==2.0.0rc4' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

# 5. Install mmdetection
git clone -b v3.0.0rc5 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
conda install "numpy<2"
pip install -v -e . --no-build-isolation
cd ..

# 6. Install mmtracking (this repo)
git clone https://github.com/khkim1729/cube.git CEUS_mmtracking
cd CEUS_mmtracking
git checkout vanilla_FiLM
pip install -r requirements/build.txt
pip install -v -e . --no-build-isolation

# 7. Additional libraries
pip install git+https://github.com/JonathonLuiten/TrackEval.git
pip uninstall -y opencv-python
pip install 'opencv-python==4.7.0.72'
```

---

## Data Preparation

### Directory Structure

```
CEUS_mmtracking/
└── data/
    └── CEUS/
        ├── annotations/
        │   └── video/
        │       ├── c1/    # ceus_video_c1_fold{k}_{split}.json
        │       ├── c2/
        │       ├── fnh/
        │       └── hcc/
        ├── Annotations/   # Raw annotation JSONs
        │   ├── post_padding.json
        │   ├── pre_padding.json
        │   └── rand_padding.json
        └── Data/          # Raw CEUS video frames
            ├── fold_0/
            ├── fold_1/
            ├── fold_2/
            ├── fold_3/
            └── fold_4/
```

### Convert to COCOVID Format

```bash
python tools/dataset_converters/ceus/ceus2coco.py
python tools/dataset_converters/ceus/fill_blank.py
```

---

## Training & Testing

### Config naming convention

```
{film_site}-{epochs}e_{dataset}_fold{k}.py
  Example: film_backbone-12e_c1_fold0.py
```

### Train

```bash
python tools/train.py \
    configs/vid/selsa_ceus_film_backbone/film_backbone-12e_c1_fold0.py \
    --work-dir experiments/film_backbone_c1_fold0
```

### Test

```bash
python tools/test.py \
    configs/vid/selsa_ceus_film_backbone/film_backbone-12e_c1_fold0.py \
    --checkpoint experiments/film_backbone_c1_fold0/best_coco_bbox_mAP_50_epoch_X.pth \
    --work-dir experiments/film_backbone_c1_fold0
```

### Run all experiments (train + test)

```bash
bash configs/vid/selsa_ceus_film_backbone/train_and_test.sh
bash configs/vid/selsa_ceus_film_neck/train_and_test.sh
bash configs/vid/selsa_ceus_film_roi/train_and_test.sh
```

### Training Plot

```bash
python tools/analysis_tools/draw_log_plots.py \
    --json results/exp_name/timestamp/vis_data/timestamp.json \
    --out_dir results/exp_name/timestamp/vis_data
```
