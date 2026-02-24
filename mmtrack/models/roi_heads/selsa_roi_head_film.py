# projects/ceus/models/selsa_roi_head_film.py

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.structures.bbox import bbox2roi
from mmtrack.registry import MODELS
from mmtrack.utils import SampleList

from .selsa_roi_head import SelsaRoIHead


@MODELS.register_module()
class SelsaRoIHeadFiLM(SelsaRoIHead):
    """SELSA RoIHead with FiLM applied right after ROIAlign outputs.

    - Applies FiLM to bbox_feats and ref_bbox_feats produced by bbox_roi_extractor
      (and after shared_head if any).
    - Key/ref 모두 적용.
    - Ref는 "ref 이미지별 proposals 개수" 기준으로 split 후, 각 ref frame phase_id로 FiLM 적용 후 concat.
    - Debug 출력은 FasterRCNNFiLM(backbone/neck) 스타일로 1회만 찍힘.
    """

    def __init__(self, *args, film_cfg: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if film_cfg is None:
            raise ValueError('film_cfg must be provided for SelsaRoIHeadFiLM.')

        # Debug (one-shot)
        self.debug = film_cfg.get('debug', False)
        self._debug_has_printed = False

        self.num_phases = film_cfg.get('num_phases', 3)
        self.emb_dim = film_cfg.get('emb_dim', 32)
        self.use_baseline = film_cfg.get('baseline', True)

        # ROIAlign(+shared_head) output channels
        self.film_channels = film_cfg.get('channels', None)
        if self.film_channels is None:
            raise ValueError('film_cfg.channels must be set (ROI feature channels).')

        self.phase_emb = nn.Embedding(self.num_phases, self.emb_dim)
        self.film_mlp = nn.Linear(self.emb_dim, 2 * self.film_channels)
        
        self._accept_ref_data_samples = True

        # init ~ identity
        nn.init.zeros_(self.film_mlp.weight)
        nn.init.zeros_(self.film_mlp.bias)

    def _get_phase_id(self, batch_data_samples: SampleList, device) -> torch.Tensor:
        """Extract phase_id from data_samples metainfo. shape: (B,)"""
        phase_ids = []
        for ds in batch_data_samples:
            if 'phase_id' not in ds.metainfo:
                raise KeyError('phase_id not found in data_sample.metainfo. '
                               'Please set it in dataset/pipeline.')
            phase_ids.append(int(ds.metainfo['phase_id']))
        return torch.tensor(phase_ids, device=device, dtype=torch.long)

    def _film_params(self, phase_id: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """phase_id: (B,) -> gamma,beta: (B,C,1,1)"""
        if phase_id.dim() == 0:
            phase_id = phase_id.view(1)
        z = self.phase_emb(phase_id)          # (B, emb_dim)
        gb = self.film_mlp(z)                 # (B, 2C)
        gamma, beta = gb.chunk(2, dim=1)      # (B, C), (B, C)
        if self.use_baseline:
            gamma = 1.0 + gamma
        gamma = gamma.view(-1, self.film_channels, 1, 1)
        beta = beta.view(-1, self.film_channels, 1, 1)
        return gamma, beta

    def _apply_film_batchwise(self, feats: Tensor, phase_id: torch.Tensor) -> Tensor:
        """feats: (N,C,H,W), phase_id: (1,) -> broadcast to all N"""
        assert feats.dim() == 4, f'Expected (N,C,H,W), got {tuple(feats.shape)}'
        assert feats.size(1) == self.film_channels, \
            f'Channel mismatch: feats has {feats.size(1)} but film_channels={self.film_channels}'

        gamma, beta = self._film_params(phase_id[:1])  # (1,C,1,1)
        return feats * gamma + beta

    def _apply_film_split_by_list(self,
                                 feats: Tensor,
                                 per_img_counts: Tuple[int, ...],
                                 phase_ids: torch.Tensor) -> Tensor:
        """Split feats by per-image proposal counts and apply per-image FiLM.
        feats: (sumK, C, H, W)
        per_img_counts: (K1, K2, ...)
        phase_ids: (num_imgs,) aligned with per_img_counts
        """
        assert len(per_img_counts) == int(phase_ids.numel()), \
            f'counts len {len(per_img_counts)} != phase_ids len {int(phase_ids.numel())}'

        chunks = list(feats.split(per_img_counts, dim=0))
        out_chunks = []
        for i, ch in enumerate(chunks):
            if ch.numel() == 0:
                out_chunks.append(ch)
                continue
            out_chunks.append(self._apply_film_batchwise(ch, phase_ids[i:i+1]))
        return torch.cat(out_chunks, dim=0) if len(out_chunks) > 0 else feats

    def _bbox_forward(self,
                      x: Tuple[Tensor],
                      ref_x: Tuple[Tensor],
                      rois: Tensor,
                      ref_rois: Tensor,
                      data_samples: Optional[SampleList] = None,
                      ref_data_samples: Optional[SampleList] = None,
                      # for correct ref split
                      rpn_results_list=None,
                      ref_rpn_results_list=None) -> dict:
        """Override bbox forward to inject FiLM right after ROIAlign outputs.

        - key: proposals는 한 장(배치1) 기준으로 전체 ROI에 key phase로 FiLM
        - ref: ref_rpn_results_list 길이 = ref frame 수. frame별 ROI 개수로 split 후, frame별 phase로 FiLM
        """
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            ref_feats=ref_x[:self.bbox_roi_extractor.num_inputs])

        ref_bbox_feats = self.bbox_roi_extractor(
            ref_x[:self.bbox_roi_extractor.num_inputs], ref_rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
            ref_bbox_feats = self.shared_head(ref_bbox_feats)

        # -----------------------
        # Debug: ROIAlign outputs
        # -----------------------
        if self.debug and not getattr(self, "_debug_has_printed", False):
            try:
                print(f"[FiLM DEBUG][roi_align:key] type={type(bbox_feats)} shape={tuple(bbox_feats.shape)} channels={int(bbox_feats.size(1))}")
                print(f"[FiLM DEBUG][roi_align:ref] type={type(ref_bbox_feats)} shape={tuple(ref_bbox_feats.shape)} channels={int(ref_bbox_feats.size(1))}")
            except Exception as e:
                print(f"[FiLM DEBUG][roi_align] failed to inspect output: {e}")

        # -------------
        # Apply FiLM key
        # -------------
        if data_samples is not None:
            key_phase = self._get_phase_id(data_samples, device=bbox_feats.device)  # (B=1,)
            if self.debug and not getattr(self, "_debug_has_printed", False):
                print("\n[FiLM DEBUG][roi:key]")
                print("phase_id:", key_phase)
                print("feat mean/std:", bbox_feats.mean().item(), bbox_feats.std().item())

            bbox_feats = self._apply_film_batchwise(bbox_feats, key_phase[:1])

            if self.debug and not getattr(self, "_debug_has_printed", False):
                print("out  mean/std:", bbox_feats.mean().item(), bbox_feats.std().item())
                print("-" * 50)

        # -------------
        # Apply FiLM ref
        # -------------
        if ref_data_samples is not None:
            ref_phase = self._get_phase_id(ref_data_samples, device=ref_bbox_feats.device)  # (Nref,)
            if self.debug and not getattr(self, "_debug_has_printed", False):
                print("\n[FiLM DEBUG][roi:ref]")
                print("phase_id:", ref_phase)
                print("feat mean/std:", ref_bbox_feats.mean().item(), ref_bbox_feats.std().item())

            # split by ref frame proposal counts if provided
            if ref_rpn_results_list is not None:
                per_ref_counts = tuple(len(p) for p in ref_rpn_results_list)
                # 안전장치: 길이 안 맞으면 fallback으로 첫 phase로 전체 broadcast
                if len(per_ref_counts) == int(ref_phase.numel()) and sum(per_ref_counts) == int(ref_bbox_feats.size(0)):
                    ref_bbox_feats = self._apply_film_split_by_list(
                        ref_bbox_feats, per_ref_counts, ref_phase)
                else:
                    ref_bbox_feats = self._apply_film_batchwise(ref_bbox_feats, ref_phase[:1])
            else:
                # fallback: 전체를 첫 ref phase로
                ref_bbox_feats = self._apply_film_batchwise(ref_bbox_feats, ref_phase[:1])

            if self.debug and not getattr(self, "_debug_has_printed", False):
                print("out  mean/std:", ref_bbox_feats.mean().item(), ref_bbox_feats.std().item())
                print("-" * 50)

        # summary line (one-shot)
        if self.debug and not getattr(self, "_debug_has_printed", False):
            print(f"[FiLM DEBUG] apply_to=roi, film_channels={self.film_channels}, "
                  f"key_phase(exists)={data_samples is not None}, ref_phase(exists)={ref_data_samples is not None}")
            print("-" * 60)
            self._debug_has_printed = True

        cls_score, bbox_pred = self.bbox_head(bbox_feats, ref_bbox_feats)
        return dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

    def bbox_loss(self,
                  x: Tuple[Tensor],
                  ref_x: Tuple[Tensor],
                  sampling_results,
                  ref_rpn_results_list,
                  data_samples: Optional[SampleList] = None,
                  ref_data_samples: Optional[SampleList] = None):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        ref_rois = bbox2roi([res.bboxes for res in ref_rpn_results_list])

        bbox_results = self._bbox_forward(
            x, ref_x, rois, ref_rois,
            data_samples=data_samples,
            ref_data_samples=ref_data_samples,
            rpn_results_list=sampling_results,              # not used for split (key is single)
            ref_rpn_results_list=ref_rpn_results_list       # used for ref split
        )

        bbox_loss_and_target = self.bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)

        bbox_results.update(loss_bbox=bbox_loss_and_target['loss_bbox'])
        return bbox_results

    def loss(self,
             x: Tuple[Tensor],
             ref_x: Tuple[Tensor],
             rpn_results_list,
             ref_rpn_results_list,
             data_samples: SampleList,
             # ✅ SELSA 쪽에서 kwargs로 넘어오면 받도록
             ref_data_samples: Optional[SampleList] = None,
             **kwargs) -> dict:
        """Training loss (override) - passes key/ref samples into bbox_loss."""
        assert len(rpn_results_list) == len(data_samples)

        batch_gt_instances = []
        batch_gt_instances_ignore = []
        for data_sample in data_samples:
            batch_gt_instances.append(data_sample.gt_instances)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)

        if self.with_bbox or self.with_mask:
            num_imgs = len(data_samples)
            sampling_results = []
            for i in range(num_imgs):
                rpn_results = rpn_results_list[i]
                rpn_results.priors = rpn_results.pop('bboxes')
                assign_result = self.bbox_assigner.assign(
                    rpn_results, batch_gt_instances[i],
                    batch_gt_instances_ignore[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    rpn_results,
                    batch_gt_instances[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()

        if self.with_bbox:
            bbox_results = self.bbox_loss(
                x, ref_x, sampling_results, ref_rpn_results_list,
                data_samples=data_samples,
                ref_data_samples=ref_data_samples
            )
            losses.update(bbox_results['loss_bbox'])

        if self.with_mask:
            bbox_feats = bbox_results['bbox_feats']
            mask_results = self.mask_loss(
                x, sampling_results, bbox_feats, batch_gt_instances)
            losses.update(mask_results['loss_mask'])

        return losses

    def predict(self,
                x: Tuple[Tensor],
                ref_x: Tuple[Tensor],
                rpn_results_list,
                ref_rpn_results_list,
                data_samples: SampleList,
                rescale: bool = False,
                # ✅ SELSA predict에서 kwargs로 받을 수 있게
                ref_data_samples: Optional[SampleList] = None,
                **kwargs):
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [ds.metainfo for ds in data_samples]

        results_list = self.predict_bbox(
            x, ref_x,
            rpn_results_list, ref_rpn_results_list,
            batch_img_metas, self.test_cfg, rescale=rescale,
            data_samples=data_samples,
            ref_data_samples=ref_data_samples
        )

        if self.with_mask:
            results_list = self.predict_mask(
                x, batch_img_metas, results_list, rescale=rescale)

        return results_list

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     ref_x: Tuple[Tensor],
                     rpn_results_list,
                     ref_rpn_results_list,
                     batch_img_metas,
                     rcnn_test_cfg,
                     rescale: bool = False,
                     data_samples: Optional[SampleList] = None,
                     ref_data_samples: Optional[SampleList] = None):
        rois = bbox2roi([res.bboxes for res in rpn_results_list])
        ref_rois = bbox2roi([res.bboxes for res in ref_rpn_results_list])

        bbox_results = self._bbox_forward(
            x, ref_x, rois, ref_rois,
            data_samples=data_samples,
            ref_data_samples=ref_data_samples,
            rpn_results_list=rpn_results_list,
            ref_rpn_results_list=ref_rpn_results_list
        )

        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in rpn_results_list)

        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(
            num_proposals_per_img, 0) if bbox_pred is not None else [None, None]

        result_list = self.bbox_head.predict_by_feat(
            rois,
            cls_score,
            bbox_pred,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)

        return result_list