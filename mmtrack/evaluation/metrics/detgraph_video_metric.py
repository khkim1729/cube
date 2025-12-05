# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, List, Tuple

import torch
from mmdet.datasets.api_wrappers import COCO
from mmdet.evaluation import CocoMetric
from mmdet.structures.mask import encode_mask_results
from mmengine.dist import broadcast_object_list, is_main_process
from mmengine.fileio import FileClient

from mmtrack.registry import METRICS
from .base_video_metrics import collect_tracking_results


@METRICS.register_module()
class DetGraphVideoMetric(CocoMetric):
    """COCO + Video-level classification metric.

    - 기존 COCO bbox mAP는 그대로 유지
    - DetGraph의 video-level logit(pred_graph_logits)과
      video_label을 이용해 classification accuracy를 추가로 계산
    """

    def __init__(self, ann_file: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file:
            file_client = FileClient.infer_client(uri=ann_file)
            with file_client.get_local_path(ann_file) as local_path:
                self._coco_api = COCO(local_path)
        else:
            self._coco_api = None

        # video-level classification (gt, pred) 쌍을 모아둘 리스트
        # 각 원소는 (gt_label:int, pred_label:int)
        self.cls_results: List[Tuple[int, int]] = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        Note:
            - detection: pred_det_instances → 기존 COCO 평가
            - classification: pred_graph_logits & video_label → accuracy
        """
        for data_sample in data_samples:
            # ---------------------------
            # 1) detection (기존 코드)
            # ---------------------------
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

            # ---------------------------
            # 2) video-level classification
            # ---------------------------
            # DetGraph.predict()에서 첫 frame에만 pred_graph_logits를 달아두었고,
            # 그 frame에 video_label도 들어 있다고 가정.
            if 'pred_graph_logits' in data_sample and 'video_label' in data_sample:
                logits = data_sample['pred_graph_logits']
                # logits: (1, num_classes) 형태의 Tensor일 것
                if isinstance(logits, torch.Tensor):
                    logits = logits.detach().cpu()
                # (1, num_classes) → scalar predicted label
                pred_label = int(logits.argmax(dim=-1).item())

                gt_label = data_sample['video_label']
                if isinstance(gt_label, torch.Tensor):
                    gt_label = int(gt_label.item())
                else:
                    gt_label = int(gt_label)

                self.cls_results.append((gt_label, pred_label))

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset.

        Returns:
            dict: {coco metrics..., 'video_cls_acc': float}
        """
        if len(self.results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.')

        # detection 결과 수집 (기존 코드)
        results = collect_tracking_results(self.results, self.collect_device)
        # classification 결과도 같은 방식으로 모아줌
        cls_results = collect_tracking_results(self.cls_results, self.collect_device)

        if is_main_process():
            # 1) COCO detection metrics
            _metrics = self.compute_metrics(results)  # type: ignore

            # 2) video-level classification accuracy
            if len(cls_results) > 0:
                correct = 0
                total = 0
                for gt_label, pred_label in cls_results:
                    total += 1
                    if gt_label == pred_label:
                        correct += 1
                video_cls_acc = correct / max(total, 1)
                _metrics['video_cls_acc'] = video_cls_acc
            else:
                # pred_graph_logits를 하나도 못 받은 경우
                _metrics['video_cls_acc'] = 0.0

            # prefix 처리
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        self.cls_results.clear()
        return metrics[0]