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
class DetGraphMeanHead(BaseModule):
    """Attention을 완전히 무시하고, 단순 mean pooling만 사용하는
    video-level graph classification head.

    Inputs:
        - node_feats: (N_nodes, C)
        - attn_weights (optional): (H, N_nodes, N_nodes)  # ← 받아도 무시

    동작:
        1) node_feats를 단순 mean pooling → graph_feat (C,)
        2) graph_feat -> MLP(FC + ReLU + Dropout + FC) -> logits
        3) loss(): CrossEntropy
        4) predict(): logits (softmax는 외부에서 필요시 적용)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 256,
        dropout: float = 0.5,
        init_cfg: OptConfigType = None,
    ) -> None:
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # 단순 MLP
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    # ----------------------------------------------------
    # 1) node-level -> graph-level pooling (mean only)
    # ----------------------------------------------------
    def _pool_graph(
        self,
        node_feats: Tensor,                   # (N, C)
        attn_weights: Optional[Tensor] = None # 인자만 받고 무시
    ) -> Tensor:
        assert node_feats.dim() == 2, \
            f'node_feats must be (N, C), got {node_feats.shape}'
        N, C = node_feats.shape

        if N == 0:
            # 안전장치: DetGraph 쪽에서 이미 필터링하지만, 혹시 몰라서
            return node_feats.new_zeros(C)

        # ★ 주의: attn_weights는 무시하고 그냥 mean
        graph_feat = node_feats.mean(dim=0)  # (C,)
        return graph_feat

    # ----------------------------------------------------
    # 2) forward: graph embedding -> logits
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
        # 1) mean pooling
        graph_feat = self._pool_graph(node_feats, attn_weights)  # (C,)

        # 2) MLP → logits
        x = self.fc1(graph_feat)
        x = self.act(x)
        x = self.drop(x)
        logits = self.fc2(x)          # (num_classes,)

        # DetGraph 인터페이스에 맞춰 (1, num_classes)
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