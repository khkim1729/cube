# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.model import BaseModule

from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType


@MODELS.register_module()
class DetGraphConcatHead(BaseModule):
    """Video-level classification head using simple concatenation of node features.

    Inputs:
        - node_feats: (N_nodes, C)
        - attn_weights (optional): (H, N_nodes, N_nodes)  # ← 무시함

    Behavior:
        1) node_feats를 (max_nodes, C)로 맞춤
           - N >= max_nodes: 상위 max_nodes만 사용
           - N < max_nodes: 남는 부분은 0-padding
        2) (max_nodes, C) → flatten → (max_nodes * C,)
        3) MLP(FC + ReLU + Dropout + FC) → logits (1, num_classes)

    이 헤드는 그래프 구조/attention을 전혀 사용하지 않고,
    단순히 "시퀀스(프레임) feature를 이어붙인 baseline" 역할을 한다.
    """

    def __init__(
        self,
        in_channels: int,           # node feature dim C
        num_classes: int,
        max_nodes: int = 16,        # CEUS: 최대 프레임 수 (T)
        hidden_channels: int = 512,
        dropout: float = 0.5,
        init_cfg: OptConfigType = None,
    ) -> None:
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # 입력 차원: max_nodes * in_channels
        concat_dim = max_nodes * in_channels

        self.fc1 = nn.Linear(concat_dim, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    # ----------------------------------------------------
    # 1) node-level -> concatenated vector
    # ----------------------------------------------------
    def _concat_nodes(
        self,
        node_feats: Tensor,                    # (N, C)
        attn_weights: Optional[Tensor] = None  # 인자만 받고 무시
    ) -> Tensor:
        assert node_feats.dim() == 2, \
            f'node_feats must be (N, C), got {node_feats.shape}'
        N, C = node_feats.shape
        assert C == self.in_channels, \
            f'in_channels mismatch: expected {self.in_channels}, got {C}'

        # (max_nodes, C) 텐서 생성
        device = node_feats.device
        out = node_feats.new_zeros(self.max_nodes, C)  # (max_nodes, C)

        if N == 0:
            # 안전장치: 모두 0인 상태로 반환
            return out.flatten(0, 1)   # (max_nodes * C,)

        # 실제 사용할 노드 수: min(N, max_nodes)
        use_n = min(N, self.max_nodes)
        # 순서는 DetGraph에서 전달된 순서(=frame 순서)를 그대로 사용
        out[:use_n, :] = node_feats[:use_n, :]

        # (max_nodes, C) -> (max_nodes * C,)
        return out.flatten(0, 1)

    # ----------------------------------------------------
    # 2) forward: concat -> logits
    # ----------------------------------------------------
    def forward(
        self,
        node_feats: Tensor,
        attn_weights: Optional[Tensor] = None
    ) -> Tensor:
        """node_feats / attn_weights를 받아 video-level logits 반환.

        Args:
            node_feats (Tensor): (N_nodes, C)
            attn_weights (Tensor, optional): (H, N_nodes, N_nodes)
                - 여기선 완전히 무시됨.

        Returns:
            logits (Tensor): (1, num_classes)
        """
        # 1) concat
        graph_feat = self._concat_nodes(node_feats, attn_weights)  # (max_nodes * C,)

        # 2) MLP
        x = self.fc1(graph_feat)
        x = self.act(x)
        x = self.drop(x)
        logits = self.fc2(x)          # (num_classes,)

        # DetGraph 인터페이스에 맞게 (1, num_classes)
        return logits.unsqueeze(0)

    # ----------------------------------------------------
    # 3) loss
    # ----------------------------------------------------
    def loss(
        self,
        node_feats: Tensor,
        labels: Tensor,
        attn_weights: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Video-level classification loss (CE).

        Args:
            node_feats (Tensor): (N_nodes, C)
            labels (Tensor): (1,) or scalar long tensor
            attn_weights (Tensor, optional): (H, N_nodes, N_nodes)

        Returns:
            dict: {'loss_ce': scalar Tensor}
        """
        logits = self.forward(node_feats, attn_weights=attn_weights)

        # labels shape 정리
        if labels.dim() == 0:
            labels = labels.view(1)
        elif labels.dim() > 1:
            labels = labels.view(-1)

        loss_ce = F.cross_entropy(logits, labels)
        return dict(loss_ce=loss_ce)

    # ----------------------------------------------------
    # 4) predict (logits만 반환)
    # ----------------------------------------------------
    def predict(
        self,
        node_feats: Tensor,
        attn_weights: Optional[Tensor] = None
    ) -> Tensor:
        """Video-level logits 반환.

        Args:
            node_feats (Tensor): (N_nodes, C)
            attn_weights (Tensor, optional): (H, N_nodes, N_nodes)

        Returns:
            logits (Tensor): (1, num_classes)
        """
        return self.forward(node_feats, attn_weights=attn_weights)