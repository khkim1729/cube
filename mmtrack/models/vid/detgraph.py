# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
from addict import Dict
from mmengine.structures import InstanceData
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import (ConfigType, OptConfigType, SampleList,
                           convert_data_sample_type)
from .base import BaseVideoDetector


@MODELS.register_module()
class DetGraph(BaseVideoDetector):
    """CEUS DetGraph
    """

    def __init__(self,
                 detector: ConfigType,
                 frozen_modules: Optional[Union[List[str], Tuple[str],
                                                str]] = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super(DetGraph, self).__init__(data_preprocessor, init_cfg)
        self.detector = MODELS.build(detector)
        assert hasattr(self.detector, 'roi_head'), \
            'selsa video detector only supports two stage detector'
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def _split_track_to_frames(self, track_sample, num_frames: int):
        """TrackDataSample 하나를 프레임 단위 DetDataSample 리스트로 쪼갠다."""
        gt = track_sample.gt_instances
        assert hasattr(gt, 'map_instances_to_img_idx'), \
            'gt_instances must have map_instances_to_img_idx for CEUS-DG.'

        idx_map = gt.map_instances_to_img_idx  # (M,)
        frame_samples = []

        for t in range(num_frames):
            mask = (idx_map == t)

            frame_gt = InstanceData()
            frame_gt.bboxes = gt.bboxes[mask]
            frame_gt.labels = gt.labels[mask]
            if hasattr(gt, 'instances_id'):
                frame_gt.instances_id = gt.instances_id[mask]

            # TrackDataSample 복제 후, 해당 프레임의 GT/메타만 남김
            frame_sample = deepcopy(track_sample)
            frame_sample.gt_instances = frame_gt

            # metainfo에서 길이 T인 것들은 t번째만 선택
            new_meta = {}
            for k, v in frame_sample.metainfo.items():
                # phase, img_id, img_path, ori_shape, img_shape, scale_factor, frame_id 등
                if isinstance(v, (list, tuple)) and len(v) == num_frames:
                    new_meta[k] = v[t]
                else:
                    # video_label, video_category, video_id, video_length 등은 그대로 유지
                    new_meta[k] = v
            frame_sample.set_metainfo(new_meta)

            frame_samples.append(frame_sample)

        return frame_samples
    
    def loss(self, inputs: dict, data_samples: SampleList, **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size and must be 1 in SELSA method.
                The T denotes the number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'DetGraph only support 1 batch size per gpu for now.'
        _, T, C, H, W = img.shape
        assert T == 16, 'DetGraph is configured for 16-frame clips currently.'

        # (1, 16, C, H, W) → (16, C, H, W)
        imgs = img.view(T, C, H, W)

        assert len(data_samples) == 1, \
            'DetGraph only supports batch_size=1 for now.'
        track_sample = data_samples[0]
        frame_samples = self._split_track_to_frames(track_sample, T)
        
        feats = self.detector.extract_feat(imgs)

        losses = dict()

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            
            rpn_data_samples = []
            # set cat_id of gt_labels to 0 in RPN
            for fs in frame_samples:
                fs_rpn = deepcopy(fs)
                fs_rpn.gt_instances.labels = torch.zeros_like(
                    fs_rpn.gt_instances.labels)
                rpn_data_samples.append(fs_rpn)
                
            rpn_losses, proposal_list = self.detector.rpn_head.loss_and_predict(
                feats, rpn_data_samples, proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = []
            for fs in frame_samples:
                proposal = InstanceData()
                proposal.bboxes = fs.proposals
                proposal_list.append(proposal)

        roi_losses = self.detector.roi_head.loss(
            feats,
            proposal_list,
            frame_samples,
            **kwargs)

        losses.update(roi_losses)

        return losses


    def predict(self,
                inputs: dict,
                data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Test without augmentation.

        Args:
            inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size and must be 1 in SELSA method.
                The T denotes the number of key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor, Optional): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            list[obj:`TrackDataSample`]: Tracking results of the
            input images. Each TrackDataSample usually contains
            ``pred_det_instances`` or ``pred_track_instances``.
        """
        img = inputs['img']
        assert img.dim() == 5, 'img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'DetGraph only supports batch_size=1 for now.'
        _, T, C, H, W = img.shape
        assert T == 16, \
            'DetGraph is currently configured for 16-frame clips.'

        # (1, 16, C, H, W) -> (16, C, H, W)
        imgs = img.view(T, C, H, W)

        # video 단위 TrackDataSample 하나가 들어옴
        assert len(data_samples) == 1, \
            'DetGraph only supports batch_size=1 for now.'
        track_sample = data_samples[0]
        frame_samples = self._split_track_to_frames(track_sample, T)
        
        feats = self.detector.extract_feat(imgs)

        if self.detector.with_rpn:
            proposal_list = self.detector.rpn_head.predict(feats, frame_samples)
            # len(proposal_list) == 16
        else:
            proposal_list = [fs.proposals for fs in frame_samples]

        results_list = self.detector.roi_head.predict(
            feats,
            proposal_list,
            frame_samples,
            rescale=rescale,
        )

        out_sample = deepcopy(track_sample)
        out_sample.pred_det_instances = results_list
        return [out_sample]

    def aug_test(self,
                 inputs: dict,
                 data_samples: SampleList,
                 rescale: bool = True,
                 **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
