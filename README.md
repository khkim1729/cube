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

### 1. conda 생성

```bash
# 1. 환경 생성
conda create -n mmlab python=3.9 -y
conda activate mmlab

# 2. pytorch, torchvision, cuda
conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch

# 3. mmengine
pip install 'mmengine==0.10.7'

# 4. mmcv
pip install 'mmcv==2.0.0rc4' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

# 5. mmdet
git clone -b v3.0.0rc5 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .

# 6. mmtrack
cd ..
git clone https://github.com/limlimlim00/CEUS_mmtracking.git
cd CEUS_mmtracking
pip install -r requirements/build.txt
pip install -v -e .

# 7. eval용 라이브러리
pip install git+https://github.com/JonathonLuiten/TrackEval.git

# 8. opencv-python version (6번에서 나온 에러: numpy와 호환되게)
pip uninstall -y opencv-python
pip install 'opencv-python==4.7.0.72'

# 9. 데모
python demo/demo_mot_vis.py configs/mot/deepsort/deepsort_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_test-mot17halfval.py —input demo/demo.mp4 —output mot.mp4
```

### 2. data 폴더 만들기
- mmtracking 하위에 data/CEUS 폴더 생성
- data/CEUS 내부에 annotations, Annotations, Data 폴더 생성
- Annotations 내부에 원본 padding json 넣기
- Data 내부에 원본 데이터 넣기

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

### 3. COCOVID json
- json을 cocovid json 형식으로 수정
- `tools/dataset_converters/ceus/ceus2coco.py` 실행 (전역변수 경로 수정) - annotations 폴더에 ceus_train.json, ceus_val.json 생성
- `tools/dataset_converters/ceus/fill_blank.py` 실행 (전역변수 경로 수정) - blank.png 생성

### 4. Configs
- configs/vid 폴더 내부에 모델 별로 config 분류
- config 이름 형식: `{모델}_{백본}_{GPU 갯수 및 batch 크기}-{epoch 수}_{데이터셋}.py`
- ex. dff_faster-rcnn_r50-dc5_1xb1-10e_ceusvid
  - 모델: dff
  - 백본: faster-rcnn_r50-dc5
  - GPU, batch: 1xb1 (GPU 1개, batch size 1)
  - epoch: 10
  - 데이터셋: ceusvid

### 5. Train
- `python tools/train.py /home/introai21/mmtracking/configs/vid/dff/dff_faster-rcnn_r50-dc5_1xb1-30e_ceusvid.py --work-dir /home/introai21/mmtracking/results/dff_r50_30e`
- 인자
  - config 경로 (필수)
  - `--work-dir`: 결과 저장 경로
- results/dff_r50_30e 과 같이 결과 저장 폴더(results) 내부에 실험 제목 폴더(dff_r50_30e) 하나 더 만들어야 함
- 내부에 timestamp 폴더가 또 생기기 때문

### 6. Training Plot
- `python tools/analysis_tools/draw_log_plots.py --json results/dff_r50_30e/20251011_174344/vis_data/20251011_174344.json --out_dir results/dff_r50_30e/20251011_174344/vis_data`
- 인자
  - `--json`: training 결과 json 경로
  - `--out_dir`: plot 저장 경로

### 7. Test & Visualization
- `python tools/test.py /home/introai21/mmtracking/configs/vid/selsa/selsa_faster-rcnn_r101-dc5_1xb8-30e_ceusapc1vid.py --checkpoint /home/introai21/mmtracking/results/selsa_r101_b8_30e_apc1/best_coco_bbox_mAP_50_epoch_12.pth --work-dir /home/introai21/mmtracking/results/selsa_r101_b8_30e_apc1`
- 인자
  - config 경로 (필수)
  - `--checkpoint`: test에 사용할 체크포인트
  - `--work-dir`: 실험 결과 저장할 폴더 (vis 폴더 내부에 시각화 결과 저장)
