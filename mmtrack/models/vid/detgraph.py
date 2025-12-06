# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from mmengine.structures import InstanceData

from mmtrack.registry import MODELS
from mmtrack.utils import (ConfigType, OptConfigType, SampleList)
from .base import BaseVideoDetector


@MODELS.register_module()
class DetGraph(BaseVideoDetector):
    """CEUS DetGraph

    - 입력: (1, T, C, H, W) 형태의 CEUS 비디오 클립 (현재 T=16 가정)
    - detector(backbone+RPN+DetGraphRoIHead)로 frame-level detection
    - DetGraphBBoxHead는 마지막 shared-FC 이후 self-attention을 수행하고,
      node_feats, attn_weights를 저장
    - DetGraph는 각 프레임별로 대표 proposal 1개씩을 선택하여
      (최대 T개 노드의) 완전연결 그래프를 구성하고,
      이를 graph_head에 넣어 video-level classification 수행
    - phase embedding이 켜져 있을 경우:
      RoIAlign 이후, BBoxHead shared FC 전에 RoI feature에 phase embedding이
      결합되며, DetGraph는 그 위에 얹힌 graph head만 다룬다.
    """

    def __init__(self,
                 detector: ConfigType,
                 graph_head: OptConfigType = None,
                 graph_loss_weight: float = 1.0,
                 frozen_modules: Optional[Union[List[str], Tuple[str], str]] = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(data_preprocessor, init_cfg)

        # ----------------------
        # BASE DETECTOR (2-stage)
        # ----------------------
        self.detector = MODELS.build(detector)
        assert hasattr(self.detector, "roi_head"), \
            "DetGraph requires a two-stage detector with a roi_head."
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # ----------------------
        # GRAPH HEAD
        # ----------------------
        self.graph_head = MODELS.build(graph_head) if graph_head else None
        self.graph_loss_weight = graph_loss_weight

        # ----------------------
        # freeze modules
        # ----------------------
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    # ----------------------------------------------------
    # TrackDataSample → frame-level DetDataSamples
    # ----------------------------------------------------
    def _split_track_to_frames(self, track_sample, num_frames: int):
        """TrackDataSample 하나를 프레임 단위로 분리.

        - gt_instances.bboxes / labels / instances_id를 frame별로 잘라서 재구성
        - metainfo 내 길이 T인 리스트(phase, phase_id, frame_id 등)는
          t번째 값만 남기도록 슬라이스
        """
        gt = track_sample.gt_instances
        assert hasattr(gt, "map_instances_to_img_idx")

        idx_map = gt.map_instances_to_img_idx  # (M,)
        frame_samples = []

        for t in range(num_frames):
            mask = (idx_map == t)

            frame_gt = InstanceData()
            frame_gt.bboxes = gt.bboxes[mask]
            frame_gt.labels = gt.labels[mask]

            if hasattr(gt, "instances_id"):
                frame_gt.instances_id = gt.instances_id[mask]

            fs = deepcopy(track_sample)
            fs.gt_instances = frame_gt

            # metainfo 중 길이가 T인 것은 t번째 것만 유지
            new_meta = {}
            for k, v in fs.metainfo.items():
                if isinstance(v, (list, tuple)) and len(v) == num_frames:
                    new_meta[k] = v[t]
                else:
                    new_meta[k] = v
            fs.set_metainfo(new_meta)

            frame_samples.append(fs)

        return frame_samples

    # ----------------------------------------------------
    # Helper: 프레임당 대표 proposal index 선택
    # ----------------------------------------------------
    def _select_one_proposal_per_frame(
        self,
        rois: Tensor,          # (N, 5)
        num_frames: int,
        cls_score: Optional[Tensor] = None,  # (N, num_classes)
        lesion_cls_idx: int = 1,
        score_thresh: float = 0.5,
    ) -> Optional[Tensor]:
        if rois is None or rois.numel() == 0:
            return None

        frame_idx = rois[:, 0].long()  # (N,)
        selected_indices = []

        # softmax로 lesion 확률 계산 (softmax 안 쓰면 그대로 logit 써도 됨)
        lesion_scores = None
        if cls_score is not None:
            probs = cls_score.softmax(dim=1)          # (N, num_classes)
            lesion_scores = probs[:, lesion_cls_idx]  # (N,)

        for t in range(num_frames):
            mask = (frame_idx == t)
            if not mask.any():
                # 이 프레임에 proposal 아예 없음 → 노드 없음
                continue

            idxs = torch.nonzero(mask, as_tuple=False).squeeze(1)  # (n_t,)

            if lesion_scores is not None:
                ls = lesion_scores[idxs]           # (n_t,)
                # threshold 이상인 proposal만 후보
                valid_mask = (ls >= score_thresh)
                if not valid_mask.any():
                    # 이 프레임에 lesion 확신있는 proposal이 없음 → 노드 없음
                    continue

                # threshold 이상 중에서 최대 lesion score 선택
                ls_valid = ls[valid_mask]                               # (k,)
                idxs_valid = idxs[valid_mask]                           # (k,)
                best_local = ls_valid.argmax()
                best_global = idxs_valid[best_local]
            else:
                # cls_score 없으면 그냥 첫 proposal 사용 (fallback)
                best_global = idxs[0]

            selected_indices.append(best_global)

        if len(selected_indices) == 0:
            return None

        return torch.stack(selected_indices, dim=0)  # (T',)  T' ≤ num_frames

    # ----------------------------------------------------
    # Helper: graph_head에 넣을 node_feats / attn_weights 생성
    # ----------------------------------------------------
    def _build_graph_inputs(
        self,
        num_frames: int
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """BBoxHead / RoIHead에 캐시된 정보를 기반으로
        프레임당 1 proposal씩 뽑아 graph_head 입력용 노드/어텐션을 구성한다.

        Returns:
            node_feats_graph (Tensor or None): (T', C)
            attn_graph (Tensor or None): (H, T', T') or None
        """
        roi_head = self.detector.roi_head
        bbox_head = roi_head.bbox_head

        # BBoxHead에서 저장한 전체 노드 임베딩 / attention
        node_feats_all = getattr(bbox_head, "last_node_feats", None)       # (N, C)
        attn_all = getattr(bbox_head, "last_attn_weights", None)           # (H, N, N)
        rois = getattr(roi_head, "last_rois", None)                        # (N, 5)

        if node_feats_all is None or rois is None:
            return None, None

        # 옵션: cls_score를 RoIHead에 저장해두었다면 사용
        cls_score_all = getattr(roi_head, "last_cls_score", None)          # (N, num_classes) or None

        # 프레임당 대표 proposal index 선택
        selected = self._select_one_proposal_per_frame(
            rois=rois,
            num_frames=num_frames,
            cls_score=cls_score_all
        )
        if selected is None:
            return None, None

        # 노드 임베딩 서브셋 (T', C)
        node_feats_graph = node_feats_all[selected]

        # 어텐션 서브셋 (H, T', T') — fully-connected 서브그래프
        if attn_all is not None:
            # attn_all: (H, N, N)
            # 선택된 index들에 대해서 행/열 모두 subset
            attn_graph = attn_all[:, selected][:, :, selected]
        else:
            attn_graph = None

        return node_feats_graph, attn_graph

    # ----------------------------------------------------
    # TRAIN
    # ----------------------------------------------------
    def loss(self, inputs: dict, data_samples: SampleList, **kwargs) -> dict:
        """Detection loss + (optional) graph loss."""
        img = inputs["img"]
        assert img.dim() == 5 and img.size(0) == 1, \
            "DetGraph expects input shape (1, T, C, H, W)."

        _, T, C, H, W = img.shape
        assert T == 16, "DetGraph currently uses T=16 frames."

        imgs = img.view(T, C, H, W)
        track_sample = data_samples[0]

        frame_samples = self._split_track_to_frames(track_sample, T)
        feats = self.detector.extract_feat(imgs)

        losses = {}

        # -----------------------------
        # 1) RPN
        # -----------------------------
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                "rpn_proposal", self.detector.test_cfg.rpn
            )

            rpn_samples = []
            for fs in frame_samples:
                fs_rpn = deepcopy(fs)
                # RPN은 class-agnostic하게 쓰고 싶어서 전부 0으로
                if fs_rpn.gt_instances.labels.numel() > 0:
                    fs_rpn.gt_instances.labels = torch.zeros_like(
                        fs_rpn.gt_instances.labels)
                rpn_samples.append(fs_rpn)

            rpn_losses, proposal_list = self.detector.rpn_head.loss_and_predict(
                feats, rpn_samples, proposal_cfg=proposal_cfg
            )
            losses.update(rpn_losses)
        else:
            proposal_list = [InstanceData(bboxes=fs.proposals)
                             for fs in frame_samples]

        # -----------------------------
        # 2) ROI HEAD LOSS
        # -----------------------------
        roi_losses = self.detector.roi_head.loss(
            feats, proposal_list, frame_samples
        )
        losses.update(roi_losses)

        # -----------------------------
        # 3) GRAPH LOSS (프레임당 1 proposal → 그래프)
        # -----------------------------
        if self.graph_head is not None:
            # 프레임당 대표 proposal들로 이루어진 노드/어텐션 추출
            node_feats_graph, attn_graph = self._build_graph_inputs(num_frames=T)

            # video-level label
            track_meta = track_sample.metainfo
            video_label = track_meta.get("video_label", None)

            if (node_feats_graph is not None) and (video_label is not None):
                # 1) 리스트/튜플인 경우 → 첫 원소만 사용 (영상당 라벨 1개라고 가정)
                if isinstance(video_label, (list, tuple)):
                    video_label = video_label[0]

                # 2) 텐서/스칼라를 (1,) 형태 long 텐서로 정규화
                if not torch.is_tensor(video_label):
                    video_label = torch.tensor(
                        [video_label],
                        dtype=torch.long,
                        device=node_feats_graph.device
                    )
                else:
                    video_label = video_label.to(node_feats_graph.device).view(1)

                graph_loss = self.graph_head.loss(
                    node_feats=node_feats_graph,
                    labels=video_label,
                    attn_weights=attn_graph
                )

                # loss 이름 prefix 추가
                for k, v in graph_loss.items():
                    losses[f"graph_{k}"] = v * self.graph_loss_weight

        return losses

    # ----------------------------------------------------
    # TEST
    # ----------------------------------------------------
    def predict(self, inputs: dict, data_samples: SampleList, rescale: bool = True):

        img = inputs["img"]
        assert img.dim() == 5 and img.size(0) == 1
        _, T, C, H, W = img.shape

        imgs = img.view(T, C, H, W)
        track_sample = data_samples[0]

        frame_samples = self._split_track_to_frames(track_sample, T)
        feats = self.detector.extract_feat(imgs)

        # proposals
        if self.detector.with_rpn:
            proposal_list = self.detector.rpn_head.predict(feats, frame_samples)
        else:
            proposal_list = [fs.proposals for fs in frame_samples]

        # detection inference
        results_list = self.detector.roi_head.predict(
            feats, proposal_list, frame_samples, rescale=rescale
        )

        out = []
        for fs, preds in zip(frame_samples, results_list):
            fs_out = deepcopy(fs)
            fs_out.pred_det_instances = preds
            out.append(fs_out)

        # ---------------------------
        # VIDEO-LEVEL GRAPH PREDICT
        # ---------------------------
        if self.graph_head is not None:
            node_feats_graph, attn_graph = self._build_graph_inputs(num_frames=T)

            if node_feats_graph is not None:
                graph_logits = self.graph_head.predict(
                    node_feats=node_feats_graph,
                    attn_weights=attn_graph
                )
                # 첫 frame에 video-level 결과를 달아둔다 (필요하면 구조 변경 가능)
                out[0].pred_graph_logits = graph_logits

        return out

    def aug_test(self, *args, **kwargs):
        raise NotImplementedError