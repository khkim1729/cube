# metrics/phase_rpn_recall.py
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS


def bbox_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU between boxes1 (N,4) and boxes2 (M,4), xyxy."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


@METRICS.register_module()
class PhaseRPNRecall(BaseMetric):
    """Phase-wise RPN recall@K plus merged PP+LP recall (no config change needed)."""

    def __init__(self, Ks=(50, 100, 300), iou_thr=0.5, **kwargs):
        super().__init__(**kwargs)
        self.Ks = list(Ks)
        self.iou_thr = float(iou_thr)
        self._stats = {}         # phase -> dict(total, hit@K)
        self._merged_stats = {}  # merged_name -> dict(total, hit@K)

        # Hard-coded merge: PP + LP
        self._merge_name = 'PP_LP'   # or 'WO'
        self._merge_set = {'PP', 'LP'}

    def _init_bucket(self, bucket: dict, key: str):
        if key not in bucket:
            bucket[key] = {'total': 0, **{f'hit@{k}': 0 for k in self.Ks}}

    def process(self, data_batch, data_samples):
        for ds in data_samples:
            # ---- read phase / props / gt ----
            if isinstance(ds, dict):
                phase = ds.get('phase', 'unknown')
                rpn_props = ds.get('rpn_props', None)

                gt_inst = ds.get('gt_instances', None)
                if gt_inst is None:
                    gt = None
                elif isinstance(gt_inst, dict):
                    gt = gt_inst.get('bboxes', None)
                else:
                    gt = getattr(gt_inst, 'bboxes', None)
            else:
                phase = ds.metainfo.get('phase', 'unknown')
                rpn_props = ds.metainfo.get('rpn_props', None)
                gt = ds.gt_instances.bboxes if hasattr(ds, 'gt_instances') else None

            # ---- validate ----
            if rpn_props is None or gt is None:
                continue
            if isinstance(gt, torch.Tensor) and gt.numel() == 0:
                continue

            if not isinstance(gt, torch.Tensor):
                gt = torch.as_tensor(gt)
            if not isinstance(rpn_props, torch.Tensor):
                rpn_props = torch.as_tensor(rpn_props, device=gt.device)

            # compute hits once
            hits = {}
            for k in self.Ks:
                topk = rpn_props[:k]
                ious = bbox_iou_xyxy(gt, topk)
                hits[k] = int((ious.max(dim=1).values >= self.iou_thr).any().item())

            # ---- per-phase accumulate ----
            self._init_bucket(self._stats, phase)
            self._stats[phase]['total'] += 1
            for k in self.Ks:
                self._stats[phase][f'hit@{k}'] += hits[k]

            # ---- merged PP+LP accumulate ----
            if phase in self._merge_set:
                self._init_bucket(self._merged_stats, self._merge_name)
                self._merged_stats[self._merge_name]['total'] += 1
                for k in self.Ks:
                    self._merged_stats[self._merge_name][f'hit@{k}'] += hits[k]

    def compute_metrics(self, results):
        out = {}

        # per-phase outputs (existing)
        for phase, st in self._stats.items():
            total = max(st['total'], 1)
            for k in self.Ks:
                out[f'RPN_recall@{k}/{phase}'] = st[f'hit@{k}'] / total

        # merged outputs (additional, always on)
        for merged_name, st in self._merged_stats.items():
            total = max(st['total'], 1)
            for k in self.Ks:
                out[f'RPN_recall@{k}/{merged_name}'] = st[f'hit@{k}'] / total

        return out