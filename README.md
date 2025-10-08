## Introduction

MMTracking is an open source video perception toolbox by PyTorch. It is a part of [OpenMMLab](https://openmmlab.com) project.

The master branch works with **PyTorch1.6+**.

## Get Started

Please refer to [get_started.md](docs/en/get_started.md) for install instructions.

Please refer to [inference.md](docs/en/user_guides/3_inference.md) for the basic usage of MMTracking. If you want to train and test your own model, please see [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md) and [train_test.md](docs/en/user_guides/4_train_test.md).

A Colab tutorial is also provided. You may preview the notebook [here](./demo/MMTracking_Tutorial.ipynb) or directly run it on [Colab](https://colab.research.google.com/github/open-mmlab/mmtracking/blob/master/demo/MMTracking_Tutorial.ipynb).

There are also usage [tutorials](docs/en/user_guides/), such as [learning about configs](docs/en/user_guides/1_config.md), [visualization](docs/en/user_guides/5_visualization.md), [analysis tools](docs/en/user_guides/6_analysis_tools.md),

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

### Video Object Detection

Supported Methods

- [x] [DFF](configs/vid/dff) (CVPR 2017)
- [x] [FGFA](configs/vid/fgfa) (ICCV 2017)
- [x] [SELSA](configs/vid/selsa) (ICCV 2019)
- [x] [Temporal RoI Align](configs/vid/temporal_roi_align) (AAAI 2021)

Supported Datasets

- [x] [ILSVRC](http://image-net.org/challenges/LSVRC/2015/)

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

## For CEUS
### 1. data 폴더 만들기
- mmtracking 하위에 data/CEUS 폴더 생성
- data/CEUS 내부에 annotations, Annotations, Data 폴더 생성
- Annotations 내부에 json, Data 내부에 원본 데이터 넣기

```
mmtracking/
├── data/
│   └── CEUS/
│       ├── annotations/
│       │
│       ├── Annotations/
│       │   ├── post_padding.json
│       │   ├── post_padding_aug.json
│       │   ├── pre_padding.json
│       │   ├── pre_padding_aug.json
│       │   ├── rand_padding.json
│       │   └── rand_padding_aug.json
│       │
│       └── Data/
│           ├── fold_0/
│           ├── fold_1/
│           ├── fold_2/
│           ├── fold_3/
│           └── fold_4/
```

### 2. COCOVID json
- json을 cocovid json 형식으로 수정
- `tools/dataset_converters/ceus/ceus2coco.py` 실행 (전역변수 경로 수정) - annotations 폴더에 ceus_train.json, ceus_val.json 생성
- `tools/dataset_converters/ceus/fill_blank.py` 실행 (전역변수 경로 수정) - blank.png 생성

### 3. Configs
- configs/vid 폴더 내부에 모델 별로 config 분류
- config 이름 형식: {모델}\_{백본}\_{GPU 갯수 및 batch 크기}-{epoch 수}\_{데이터셋}.py
- ex. dff_faster-rcnn_r50-dc5_1xb1-10e_ceusvid
  - 모델: dff
  - 백본: faster-rcnn_r50-dc5
  - GPU, batch: 1xb1 (GPU 1개, batch size 1)
  - epoch: 10
  - 데이터셋: ceusvid

### 4. Train 돌리기
- `python tools/train.py /home/introai21/mmtracking/configs/vid/dff/dff_faster-rcnn_r50-dc5_1xb1-10e_ceusvid.py --work-dir /home/introai21/mmtracking/results/dff_example`
- 인자: config 경로, 결과 저장 경로
- results/dff_example 과 같이 결과 저장 폴더(results) 내부에 실험 제목 폴더(dff_example) 하나 더 만들어야 함

### 5. ToDo
- 10. 8.
  - 구현 완료
    - DFF (num_classes = 2, num_frames = 16)
      - `dff_faster-rcnn_r50-dc5_1xb1-10e_ceusvid.py`
      - `dff_faster-rcnn_r101-dc5_1xb1-10e_ceusvid.py`
      - `dff_faster-rcnn_x101-dc5_1xb1-10e_ceusvid.py`
    - FGFA (num_classes = 2, num_frames = 16)
      - `fgfa_faster-rcnn_r50-dc5_1xb1-10e_ceusvid.py`
      - `fgfa_faster-rcnn_r101-dc5_1xb1-10e_ceusvid.py`
      - `fgfa_faster-rcnn_x101-dc5_1xb1-10e_ceusvid.py`
  - 구현 예정
    - DFF (num_classes = 1)
    - FGFA (num_classes = 1)
    