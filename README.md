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
- mmtracking 바로 밑에 data 라는 폴더를 만들어줘야 함.
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
- 우리 json을 cocovid json 형식으로 바꿔줘야함.
- 이를 위해 tools/dataset_converters/ceus/ceus2coco.py를 실행한다. 이때, 전역변수에 있는 경로는 꼭 맞춰주기!
  - annotations 아래에 ceus_train.json과 ceus_val.json 이 생긴 것을 확인할 수 있다.
- 이후 tools/dataset_converters/ceus/fill_blank.py를 실행해서, blank.png를 만들어주자. 역시 경로 맞춰주고 실행.


### 3. Configs
- vid/dff: 이 디렉토리 내부가 최종 config. train 돌릴 때 이 경로를 넣어주면 된다.
- config 이름 형식: {모델}_{백본}_{GPU 갯수 및 batch 갯수 (ex. GPU 1개, 배치 1: 1xb1)}-{epoch 수}_{데이터셋}.py

### 4. Train 돌리기
- python tools/train.py /home/introai21/mmtracking/configs/vid/dff/dff_faster-rcnn_r50-dc5_1xb1-10e_ceusvid.py --work-dir /home/introai21/mmtracking/results/dff_example
- config, 결과 저장 경로 차례로 넣어주면 된다. results/dff_example 과 같이 결과 저장 폴더 내부에 실험 제목 폴더 하나 더 만들어야 안섞임.
