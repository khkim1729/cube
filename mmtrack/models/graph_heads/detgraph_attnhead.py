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
class DetGraphAttnHead(BaseModule):
    """Video-level graph classification head for DetGraph.

    Inputs:
        - node_feats: (N_nodes, C)
        - attn_weights (optional): (H, N_nodes, N_nodes)

    Behavior:
        1) node_feats를 graph embedding 하나로 pooling
           - attn_weights가 있으면, attention 기반 가중 합
           - 없으면 단순 mean pooling
        2) graph embedding -> MLP -> logits
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

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    # ----------------------------------------------------
    # 1) node-level -> graph-level pooling
    # ----------------------------------------------------
    def _pool_graph(
        self,
        node_feats: Tensor,          # (N, C)
        attn_weights: Optional[Tensor] = None  # (H, N, N) or None
    ) -> Tensor:
        """노드 임베딩을 그래프 임베딩 하나로 요약.

        - attn_weights가 있으면:
            각 노드의 "중요도"를 attention 기반으로 구해
            가중합 pooling 수행
        - 없으면:
            단순 mean pooling
        """
        assert node_feats.dim() == 2, \
            f'node_feats must be (N, C), got {node_feats.shape}'
        N, C = node_feats.shape

        if N == 0:
            # 안전장치: 이 경우는 DetGraph에서 이미 걸러주지만,
            # 혹시 몰라 zero vector 반환
            return node_feats.new_zeros(C)

        if attn_weights is None:
            # (N, C) -> (C,)
            graph_feat = node_feats.mean(dim=0)
            return graph_feat

        # attn_weights: (H, N, N)
        # head 평균
        attn_mean = attn_weights.mean(dim=0)  # (N, N)

        # 각 노드의 "importance"를 attention 기반으로 계산
        # - incoming + outgoing attention을 모두 고려
        #   importance_i = 평균_j attn(i->j) + 평균_j attn(j->i)
        out_importance = attn_mean.mean(dim=1)  # (N,)
        in_importance = attn_mean.mean(dim=0)   # (N,)
        importance = (out_importance + in_importance) * 0.5  # (N,)

        # softmax로 normalize (N,)
        weights = F.softmax(importance, dim=0).unsqueeze(1)  # (N, 1)

        # 가중합 pooling
        graph_feat = (weights * node_feats).sum(dim=0)       # (C,)

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

        Returns:
            logits (Tensor): (1, num_classes)
        """
        # 1) node-level -> graph-level pooling
        graph_feat = self._pool_graph(node_feats, attn_weights)  # (C,)

        # 2) MLP -> logits
        x = self.fc1(graph_feat)
        x = self.act(x)
        x = self.drop(x)
        logits = self.fc2(x)        # (num_classes,)

        # DetGraph쪽 인터페이스 맞추기 위해 (1, num_classes)로 reshape
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
        """Video-level classification loss.

        Args:
            node_feats (Tensor): (N_nodes, C)
            labels (Tensor): (1,) or scalar long tensor
            attn_weights (Tensor, optional): (H, N_nodes, N_nodes)

        Returns:
            dict: {'loss_ce': scalar Tensor}
        """
        logits = self.forward(node_feats, attn_weights=attn_weights)  # (1, num_classes)

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
        logits = self.forward(node_feats, attn_weights=attn_weights)
        return logits