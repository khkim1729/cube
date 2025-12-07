# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence

from mmdet.datasets.api_wrappers import COCO
from mmdet.evaluation import CocoMetric
from mmdet.structures.mask import encode_mask_results
from mmengine.dist import broadcast_object_list, is_main_process
from mmengine.fileio import FileClient

from mmtrack.registry import METRICS
from .base_video_metrics import collect_tracking_results


@METRICS.register_module()
class DetGraphVideoMetric(CocoMetric):
    """DetGraph용 COCO + video-level classification metric.

    - bbox mAP: CocoVideoMetric과 동일 (pred_det_instances 사용)
    - video-level cls: pred_graph_logits vs video_label 기반 accuracy 계산

    요구사항:
      - 각 frame data_sample:
          pred_det_instances: detection 결과
          ori_shape, img_id : GT 파싱에 사용
      - DetGraph.predict()에서
          첫 프레임 data_sample에 pred_graph_logits를 심어둔다고 가정
      - GT video label:
          data_sample['video_label']에 존재한다고 가정
    """

    def __init__(self, ann_file: Optional[str] = None, **kwargs) -> None:
        # ★ CocoVideoMetric과 동일하게 ann_file은 super에 전달하지 않음
        super().__init__(**kwargs)

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file:
            file_client = FileClient.infer_client(uri=ann_file)
            with file_client.get_local_path(ann_file) as local_path:
                self._coco_api = COCO(local_path)
        else:
            self._coco_api = None

        # ★ 추가: video-level cls 결과 저장용
        # 각 원소: dict(video_id=..., frame_id=..., pred_label=int, gt_label=int)
        self.video_results = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """한 batch에 대해
        - bbox 결과는 CocoVideoMetric처럼 수집
        - video-level cls 결과는 별도 리스트에 수집
        """
        for data_sample in data_samples:
            # -----------------------------
            # (A) bbox 결과 (기존 CocoVideoMetric 그대로)
            # -----------------------------
            result = dict()
            pred = data_sample['pred_det_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy())
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))

            # -----------------------------
            # (B) video-level classification 결과 (추가 부분)
            # -----------------------------
            # DetGraph.predict()에서 out[0]에만 pred_graph_logits를 넣었으니
            # 자연스럽게 "첫 프레임"만 여기로 들어옴.
            if 'pred_graph_logits' not in data_sample:
                continue

            logits = data_sample['pred_graph_logits']
            if hasattr(logits, 'detach'):
                logits = logits.detach().cpu()

            # (1, C) → (C,)
            if logits.dim() == 2:
                logits = logits.squeeze(0)

            pred_label = int(logits.argmax(dim=-1).item())

            # GT video label
            video_label = data_sample.get('video_label', None)
            if video_label is None:
                continue

            # 리스트/텐서/int 등 케이스 안전 처리
            if isinstance(video_label, (list, tuple)):
                if len(video_label) == 0:
                    continue
                video_label = video_label[0]

            if hasattr(video_label, 'item'):
                gt_label = int(video_label.item())
            else:
                gt_label = int(video_label)

            video_id = data_sample.get('video_id', None)
            frame_id = data_sample.get('frame_id', None)

            self.video_results.append(
                dict(
                    video_id=video_id,
                    frame_id=frame_id,
                    pred_label=pred_label,
                    gt_label=gt_label)
            )

    def evaluate(self, size: int) -> dict:
        """bbox mAP (COCO) + video-level accuracy를 함께 리턴."""
        if len(self.results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.')

        # rank 별 results 모으기
        results = collect_tracking_results(self.results, self.collect_device)
        video_results = collect_tracking_results(
            self.video_results, self.collect_device)

        if is_main_process():
            # (1) COCO bbox metrics (기존과 동일)
            _metrics = self.compute_metrics(results)  # type: ignore

            # (2) video-level classification accuracy (추가)
            if video_results is not None and len(video_results) > 0:
                preds = [int(r['pred_label']) for r in video_results]
                gts = [int(r['gt_label']) for r in video_results]
                assert len(preds) == len(gts)
                correct = sum(p == g for p, g in zip(preds, gts))
                acc = correct / max(len(gts), 1)
                _metrics['video_cls_acc'] = float(acc)

            # prefix 적용 (coco/...)
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset
        self.results.clear()
        self.video_results.clear()

        return metrics[0]