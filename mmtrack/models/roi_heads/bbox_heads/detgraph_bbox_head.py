from typing import Tuple, Optional

import torch.nn as nn
from mmdet.models import ConvFCBBoxHead
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import ConfigType


@MODELS.register_module()
class DetGraphBBoxHead(ConvFCBBoxHead):
    """DetGraph용 BBox Head.

    - 기본 구조는 ConvFCBBoxHead와 동일
    - shared FC 블록 이후에 self-attention 기반 Aggregator 한 번 태움
    - ref_x 없이, (N, C) proposal feature들끼리 self-attention

    Args:
        aggregator (ConfigType): DetGraphAggregator 설정.
            예시:
            aggregator = dict(
                type='DetGraphAggregator',
                in_channels=1024,
                num_heads=8,
            )
    """

    def __init__(self, aggregator: Optional[ConfigType] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # aggregator가 주어졌고 shared FC가 있을 때만 사용
        self.use_aggregator = aggregator is not None and self.num_shared_fcs > 0

        if self.use_aggregator:
            # 하나의 aggregator를 shared_fcs 블록 끝에서 사용
            self.aggregator = MODELS.build(aggregator)
        else:
            self.aggregator = None

        # SELSA 쪽에서 쓰던 것과 동일하게 inplace=False ReLU 하나 둠
        self.inplace_false_relu = nn.ReLU(inplace=False)

        # graph-level에서 재사용할 attention / node feats 저장 공간
        # forward() 호출 시마다 갱신됨
        self.last_attn_weights = None  # (H, N, N) or None
        self.last_node_feats = None    # (N, C) or None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """RoI feature로부터 cls_score / bbox_pred 계산.

        Args:
            x (Tensor): shape [N, C, H, W] 또는 [N, C]

        Returns:
            tuple:
                - cls_score (Tensor): (N, num_classes) or None
                - bbox_pred (Tensor): (N, 4 * num_classes) or (N, 4) or None
        """
        # -------------------------
        # 1) shared convs
        # -------------------------
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        # -------------------------
        # 2) shared FCs + aggregator
        # -------------------------
        if self.num_shared_fcs > 0:
            # ConvFCBBoxHead 기본 구조와 동일
            if self.with_avg_pool and x.dim() > 2:
                x = self.avg_pool(x)

            # [N, C, H, W] or [N, C] -> [N, C']
            x = x.flatten(1)

            # shared FC 통과
            for fc in self.shared_fcs:
                x = fc(x)
                x = self.inplace_false_relu(x)

            # 여기서 self-attention aggregator 한 번 적용
            if self.use_aggregator:
                # DetGraphAggregator는 (N, C) -> (N, C), (H, N, N)
                x, attn = self.aggregator(x)
                self.last_attn_weights = attn
            else:
                # aggregator를 쓰지 않는 경우에도 graph_head에서
                # 사용할 수 있도록 post-FC feature를 node_feats로 저장
                self.last_attn_weights = None

            # graph용 node feature는 aggregator 적용 직후의 x로 통일
            self.last_node_feats = x  # (N, C)

            # 이후 cls/reg branch에 들어가기 전 ReLU
            x = self.inplace_false_relu(x)

        # -------------------------
        # 3) cls / reg 분기
        # -------------------------
        x_cls = x
        x_reg = x

        # cls 분기
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        # reg 분기
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # 최종 헤드
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred