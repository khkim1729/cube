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
    """Phase-wise RPN recall@K: hit if any proposal in top-K covers any GT with IoU>=thr."""
    def __init__(self, Ks=(50, 100, 300), iou_thr=0.5, **kwargs):
        super().__init__(**kwargs)
        self.Ks = list(Ks)
        self.iou_thr = float(iou_thr)
        self._stats = {}  # phase -> dict(total, hit@K)

    def process(self, data_batch, data_samples):
        # In mmtrack test loop, outputs can be list[dict]
        for ds in data_samples:
            # ---- phase ----
            if isinstance(ds, dict):
                phase = ds.get('phase', 'unknown')
                rpn_props = ds.get('rpn_props', None)

                gt_inst = ds.get('gt_instances', None)
                if gt_inst is None:
                    gt = None
                elif isinstance(gt_inst, dict):
                    gt = gt_inst.get('bboxes', None)
                else:
                    # InstanceData
                    gt = getattr(gt_inst, 'bboxes', None)

            else:
                # fallback if you ever get TrackDataSample
                phase = ds.metainfo.get('phase', 'unknown')
                rpn_props = ds.metainfo.get('rpn_props', None)
                gt = ds.gt_instances.bboxes if hasattr(ds, 'gt_instances') else None

            # ---- validate ----
            if rpn_props is None:
                continue
            if gt is None:
                continue
            if isinstance(gt, torch.Tensor) and gt.numel() == 0:
                continue

            # tensors
            if not isinstance(gt, torch.Tensor):
                gt = torch.as_tensor(gt)
            if not isinstance(rpn_props, torch.Tensor):
                rpn_props = torch.as_tensor(rpn_props, device=gt.device)

            # init phase stats
            if phase not in self._stats:
                self._stats[phase] = {'total': 0, **{f'hit@{k}': 0 for k in self.Ks}}

            self._stats[phase]['total'] += 1

            # recall@K
            for k in self.Ks:
                topk = rpn_props[:k]
                ious = bbox_iou_xyxy(gt, topk)  # (G,K)
                hit = (ious.max(dim=1).values >= self.iou_thr).any().item()
                self._stats[phase][f'hit@{k}'] += int(hit)

    def compute_metrics(self, results):
        out = {}
        for phase, st in self._stats.items():
            total = max(st['total'], 1)
            for k in self.Ks:
                out[f'RPN_recall@{k}/{phase}'] = st[f'hit@{k}'] / total
        return out