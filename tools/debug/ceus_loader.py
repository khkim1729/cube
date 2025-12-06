# debug_ceus_loader.py
import pprint

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmtrack.utils import register_all_modules
from mmengine.config import Config
from mmtrack.registry import DATASETS

def main():
    register_all_modules(init_default_scope=True)
    # 1. config 불러오기
    cfg = Config.fromfile('configs/vid/dg/dg_faster-rcnn_r50-dc5_1xb1-30e_ceusdgvid.py')  # 너가 쓰는 cfg 경로

    # 2. train dataset 빌드
    train_dataset_cfg = cfg.train_dataloader['dataset']
    train_dataset = DATASETS.build(train_dataset_cfg)

    print(f'len(train_dataset) = {len(train_dataset)}')

    # 3. 샘플 하나 뽑기 (pipeline + PackTrackInputs 까지 다 적용된 상태)
    sample = train_dataset[0]

    # ---- inputs 확인 ----
    inputs = sample['inputs']
    data_sample = sample['data_samples']

    print('\n[inputs]')
    print('img shape:', inputs['img'].shape)  # 기대: torch.Size([16, 3, 512, 512])

    # ref_img가 없어야 함 (없으면 KeyError 날 수 있으니 안전하게)
    print('has ref_img?', 'ref_img' in inputs)

    # ---- metainfo 확인 ----
    print('\n[data_samples.metainfo keys]')
    pprint.pprint(data_sample.metainfo.keys())

    print('\nvideo_label:', data_sample.metainfo.get('video_label', None))
    print('video_category:', data_sample.metainfo.get('video_category', None))
    print('frame_id:', data_sample.metainfo.get('frame_id', None))
    print('phase:', data_sample.metainfo.get('phase', None))

    # ---- gt_instances 확인 ----
    inst = data_sample.gt_instances
    print('\n[gt_instances]')
    print('bboxes shape:', getattr(inst, 'bboxes', None).shape if hasattr(inst, 'bboxes') else None)
    print('labels shape:', getattr(inst, 'labels', None).shape if hasattr(inst, 'labels') else None)
    print('instances_id shape:', getattr(inst, 'instances_id', None).shape if hasattr(inst, 'instances_id') else None)
    print('map_instances_to_img_idx:', getattr(inst, 'map_instances_to_img_idx', None))

if __name__ == '__main__':
    main()