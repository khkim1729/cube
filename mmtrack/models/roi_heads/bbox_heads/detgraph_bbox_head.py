# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Optional

import torch
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
    - (옵션) phase embedding을 proposal feature에 주입

    Args:
        aggregator (ConfigType): DetGraphAggregator 설정.
            예시:
            aggregator = dict(
                type='DetGraphAggregator',
                in_channels=1024,
                num_attention_blocks=8,
            )

        use_phase_embed (bool): phase embedding 사용 여부.
        num_phases (int): phase 클래스 개수 (AP/PP/LP/KP 등을 묶은 수).
        phase_embed_dim (int): phase embedding 차원.
        phase_fusion_mode (str): 'concat' 또는 'add'.
            - 'concat': [x, phase_emb] concat 후 Linear로 fc_out_channels 로 투영
            - 'add':    x와 phase_emb를 같은 차원에서 elementwise add
        unknown_phase_id (int): 유효하지 않은 phase (예: -1) 표시용 id.
    """

    def __init__(
        self,
        aggregator: Optional[ConfigType] = None,
        use_phase_embed: bool = False,
        num_phases: int = 3,
        phase_embed_dim: int = 32,
        phase_fusion_mode: str = 'concat',  # or 'add'
        unknown_phase_id: int = -1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # -----------------------------
        # 0) Phase Embedding 설정
        # -----------------------------
        self.use_phase_embed = use_phase_embed
        self.num_phases = num_phases
        self.phase_embed_dim = phase_embed_dim
        self.phase_fusion_mode = phase_fusion_mode
        self.unknown_phase_id = unknown_phase_id

        if self.use_phase_embed:
            assert self.phase_fusion_mode in ('concat', 'add'), \
                f'phase_fusion_mode must be "concat" or "add", got {self.phase_fusion_mode}'

            # phase_id ∈ [0, num_phases-1] 를 위한 embedding
            self.phase_embed = nn.Embedding(num_phases, phase_embed_dim)

            if self.phase_fusion_mode == 'concat':
                # shared_fcs 출력 차원은 self.fc_out_channels
                # concat 후 다시 fc_out_channels로 투영해서
                # 기존 aggregator / cls / reg 구조 안 깨지게 유지
                self.phase_fuse = nn.Linear(
                    self.fc_out_channels + phase_embed_dim,
                    self.fc_out_channels
                )
            else:
                # 'add' 모드: 동일 차원에서 더해야 하므로 dim 일치 필요
                assert phase_embed_dim == self.fc_out_channels, \
                    f'add fusion requires phase_embed_dim({phase_embed_dim}) ' \
                    f'== fc_out_channels({self.fc_out_channels})'
                self.phase_fuse = None
        else:
            self.phase_embed = None
            self.phase_fuse = None

        # -----------------------------
        # 1) Aggregator 설정
        # -----------------------------
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

    # -------------------------------------------------
    # forward
    # -------------------------------------------------
    def forward(
        self,
        x: Tensor,
        phase_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """RoI feature로부터 cls_score / bbox_pred 계산.

        Args:
            x (Tensor): shape [N, C, H, W] 또는 [N, C]
            phase_ids (Tensor, optional): (N,)
                - 각 RoI가 속한 frame의 phase_id
                - 0~num_phases-1: 유효 phase
                - unknown_phase_id (기본 -1): invalid (embedding 안 쓰거나 zero로 처리)

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
        # 2) shared FCs + (phase embed) + aggregator
        # -------------------------
        if self.num_shared_fcs > 0:
            # ConvFCBBoxHead 기본 구조와 동일
            if self.with_avg_pool and x.dim() > 2:
                x = self.avg_pool(x)

            # [N, C, H, W] or [N, C] -> [N, C']
            x = x.flatten(1)  # (N, F)

            # shared FC 통과
            for fc in self.shared_fcs:
                x = fc(x)
                x = self.inplace_false_relu(x)  # (N, fc_out_channels)

            # -------------------------
            # 2-1) Phase embedding 주입
            # -------------------------
            if self.use_phase_embed and (phase_ids is not None):
                # phase_ids: (N,)
                # unknown_phase_id (ex: -1)는 embedding 안 쓰거나 zero로 처리
                valid = (phase_ids >= 0) & (phase_ids < self.num_phases)
                if valid.any():
                    # invalid index는 일단 0으로 클램프 후 나중에 zeroing
                    phase_ids_clamped = phase_ids.clone()
                    phase_ids_clamped[~valid] = 0
                    phase_emb = self.phase_embed(phase_ids_clamped)  # (N, D)

                    # invalid 위치는 0으로
                    phase_emb = phase_emb * valid.unsqueeze(1).to(phase_emb.dtype)

                    if self.phase_fusion_mode == 'concat':
                        # [x, phase_emb] concat 후 projection
                        x = torch.cat([x, phase_emb], dim=1)  # (N, F + D)
                        x = self.phase_fuse(x)                # (N, F)
                    else:  # 'add'
                        x = x + phase_emb
                # valid 하나도 없으면 phase 정보는 그냥 스킵

            # -------------------------
            # 2-2) self-attention aggregator (SELSA-style residual)
            # -------------------------
            if self.use_aggregator:
                # DetGraphAggregator: (N, C) -> (N, C), (H, N, N)
                agg_out, attn = self.aggregator(x)
                self.last_attn_weights = attn

                # SELSA와 동일한 residual 패턴: x = x + agg(x, ref_x)
                x_res = x + agg_out          # (N, C)

                # graph용 node feature는 residual 이후의 feature로 저장
                self.last_node_feats = x_res

                # 이후 cls/reg branch에 들어가기 전 ReLU
                x = self.inplace_false_relu(x_res)
            else:
                # aggregator를 쓰지 않는 경우: shared_fcs 출력 그대로 사용
                self.last_attn_weights = None
                self.last_node_feats = x
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