# mmtrack/evaluation/detgraph_video_metric.py 이런 식으로 두면 좋음

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, List, Tuple

import torch
from mmengine.dist import broadcast_object_list, is_main_process

from mmtrack.registry import METRICS
from .coco_video_metric import CocoVideoMetric  # 🔴 네가 쓰던 원본 CocoVideoMetric

@METRICS.register_module()
class DetGraphVideoMetric(CocoVideoMetric):
    """CocoVideoMetric + Video-level classification metric.

    - COCO bbox mAP는 CocoVideoMetric 그대로 사용
    - DetGraph의 video-level logit(pred_graph_logits)과
      video_label로 classification accuracy를 추가 계산
    """

    def __init__(self,
                 ann_file: Optional[str] = None,
                 **kwargs) -> None:
        # 🔴 detection 쪽 설정은 CocoVideoMetric에게 그대로 맡김
        super().__init__(ann_file=ann_file, **kwargs)

        # video-level classification (gt, pred) 쌍을 모아둘 리스트
        # 각 원소는 (gt_label:int, pred_label:int)
        self.cls_results: List[Tuple[int, int]] = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """한 배치에 대해:
        1) detection → 부모(CocoVideoMetric)의 process 호출
        2) video-level cls → 별도로 gt/pred 쌍만 모아둔다
        """
        # 1) detection은 원본 CocoVideoMetric이 하던 그대로
        super().process(data_batch, data_samples)

        # 2) video-level classification 정보만 추가 수집
        for data_sample in data_samples:
            if 'pred_graph_logits' in data_sample and 'video_label' in data_sample:
                logits = data_sample['pred_graph_logits']
                if isinstance(logits, torch.Tensor):
                    logits = logits.detach().cpu()
                pred_label = int(logits.argmax(dim=-1).item())

                gt_label = data_sample['video_label']
                if isinstance(gt_label, torch.Tensor):
                    gt_label = int(gt_label.item())
                else:
                    gt_label = int(gt_label)

                self.cls_results.append((gt_label, pred_label))

    def evaluate(self, size: int) -> dict:
        """Evaluate 전체 데이터셋.

        - 우선 CocoVideoMetric.evaluate로 bbox mAP들 계산
        - 그 위에 video_cls_acc를 하나 추가
        """
        # 1) detection metric 먼저 계산 (bbox_mAP_* 다 여기서 세팅됨)
        _metrics = super().evaluate(size)

        # 분산 환경일 경우 video cls 결과도 모아야 함
        if len(self.cls_results) > 0:
            results_list = [self.cls_results]
            broadcast_object_list(results_list)
            gathered_cls = results_list[0]
        else:
            gathered_cls = []

        # 2) video-level classification accuracy
        if is_main_process():
            if len(gathered_cls) > 0:
                correct = 0
                total = 0
                for gt_label, pred_label in gathered_cls:
                    total += 1
                    if gt_label == pred_label:
                        correct += 1
                video_cls_acc = correct / max(total, 1)
            else:
                video_cls_acc = 0.0

            # prefix 붙이기 (CocoVideoMetric와 동일하게 self.prefix 사용)
            key_name = 'video_cls_acc'
            if self.prefix:
                key_name = '/'.join((self.prefix, key_name))
            _metrics[key_name] = video_cls_acc

        # 로컬 상태 리셋
        self.cls_results.clear()
        return _metrics