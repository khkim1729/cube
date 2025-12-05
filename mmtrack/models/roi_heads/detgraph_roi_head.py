# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from mmdet.models import StandardRoIHead
from mmdet.structures.bbox import bbox2roi
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import ConfigType, InstanceList, SampleList


@MODELS.register_module()
class DetGraphRoIHead(StandardRoIHead):

    def loss(self,
             x: Tuple[Tensor],
             rpn_results_list: InstanceList,
             data_samples: SampleList) -> dict:

        assert len(rpn_results_list) == len(data_samples)

        batch_gt_instances = []
        batch_gt_ignore = []
        for s in data_samples:
            batch_gt_instances.append(s.gt_instances)
            batch_gt_ignore.append(getattr(s, "ignored_instances", None))

        # --------------------------
        # assign + sample
        # --------------------------
        sampling_results = []
        for i in range(len(data_samples)):
            rpn_res = rpn_results_list[i]
            # mmdet는 rpn_res.bboxes로 priors를 사용
            rpn_res.priors = rpn_res.pop("bboxes")

            assign_res = self.bbox_assigner.assign(
                rpn_res, batch_gt_instances[i], batch_gt_ignore[i]
            )
            sample_res = self.bbox_sampler.sample(
                assign_res, rpn_res, batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x]
            )
            sampling_results.append(sample_res)

        # --------------------------
        # BBox head loss
        # --------------------------
        bbox_results = self.bbox_loss(x, sampling_results)

        losses = dict()
        # StandardRoIHead 스타일: loss_cls, loss_bbox는 각각 dict
        if 'loss_cls' in bbox_results:
            losses.update(bbox_results['loss_cls'])
        if 'loss_bbox' in bbox_results:
            losses.update(bbox_results['loss_bbox'])

        # graph-level을 위한 정보는 loss dict에 넣지 않고,
        # RoIHead 내부 attribute로만 보관
        self.last_bbox_feats = bbox_results["bbox_feats"]  # (N, C, H, W) 또는 (N, C)
        self.last_cls_score = bbox_results["cls_score"]
        self.last_attn_weights = self.bbox_head.last_attn_weights  # (H, N, N) or None
        self.last_rois = bbox2roi([r.bboxes for r in sampling_results])  # (N, 5)

        return losses

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor) -> dict:

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois
        )

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        return dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats
        )

    def bbox_loss(self, x: Tuple[Tensor], sampling_results):

        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        # 여기서 loss_and_target이 반환하는 dict 전체를 받아서 merge
        # 보통 {'loss_cls': {...}, 'loss_bbox': {...}, 'bbox_targets': ...} 형태
        bbox_head_outs = self.bbox_head.loss_and_target(
            cls_score=bbox_results["cls_score"],
            bbox_pred=bbox_results["bbox_pred"],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg
        )

        bbox_results.update(bbox_head_outs)
        return bbox_results

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                data_samples: SampleList,
                rescale: bool = False):

        rois = bbox2roi([res.bboxes for res in rpn_results_list])
        bbox_results = self._bbox_forward(x, rois)

        # loss()에서 쓰는 것과 동일하게 캐시
        self.last_rois = rois
        self.last_cls_score = bbox_results["cls_score"]

        # graph-level에 사용할 정보도 Test 시에는 data_samples에 심어둠
        for ds in data_samples:
            ds.bbox_feats = bbox_results["bbox_feats"]
            ds.attn_weights = self.bbox_head.last_attn_weights
            ds.roi_batch_idx = rois[:, 0]

        # 각 frame 별 proposal 개수로 split
        num_proposals_per_img = [len(p) for p in rpn_results_list]

        results = self.bbox_head.predict_by_feat(
            rois=rois.split(num_proposals_per_img, 0),
            cls_scores=bbox_results["cls_score"].split(
                num_proposals_per_img, 0),
            bbox_preds=(bbox_results["bbox_pred"].split(
                num_proposals_per_img, 0)
                if bbox_results["bbox_pred"] is not None else None),
            batch_img_metas=[ds.metainfo for ds in data_samples],
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale
        )

        return results