from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.model import BaseModule

from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType


class SimpleGCNLayer(nn.Module):
    """한 층짜리 GCN: H' = σ(Â H W).

    - x: (N, C_in)
    - adj: (N, N)  (row-normalized adjacency)
    - out: (N, C_out)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        # 1) 선형 변환
        x = self.lin(x)          # (N, C_out)
        # 2) 인접 행렬을 통한 aggregation
        x = adj @ x              # (N, C_out)
        # 3) 비선형
        return F.relu(x)


@MODELS.register_module()
class DetGraphGCNHead(BaseModule):
    """GCN + global pooling 기반 video-level graph head (PyG 없이 구현).

    Inputs:
        - node_feats: (N_nodes, C)
        - attn_weights (optional): (H, N_nodes, N_nodes)

    동작:
        1) attn_weights → soft adjacency (N,N)
        2) GCN layer stack(SimpleGCNLayer) 돌림
        3) global mean pooling → graph_feat
        4) FC → logits (1, num_classes)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 256,
        gcn_layers: int = 2,
        dropout: float = 0.5,
        init_cfg: OptConfigType = None,
    ) -> None:
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.gcn_layers = gcn_layers
        self.dropout = dropout

        # -----------------------------
        # GCN stack (SimpleGCNLayer)
        # -----------------------------
        gcn_list = []
        in_c = in_channels
        for _ in range(gcn_layers):
            gcn_list.append(SimpleGCNLayer(in_c, hidden_channels))
            in_c = hidden_channels
        self.gcn = nn.ModuleList(gcn_list)

        # -----------------------------
        # Readout + classifier
        # -----------------------------
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.drop = nn.Dropout(dropout)

    # ----------------------------------------------------
    # attn_weights → adjacency (row-normalized)
    # ----------------------------------------------------
    def _build_adj(
        self,
        attn_weights: Optional[Tensor],
        N: int,
        device: torch.device
    ) -> Tensor:
        """attn_weights(H,N,N)를 adjacency(N,N)로 변환.

        - attn 없으면: fully-connected uniform 그래프 (self-loop 포함)
        - attn 있으면:
            * head 평균 → (N,N)
            * 대칭화 (i->j, j->i 반영)
            * self-loop 추가
            * row-normalization
        """
        if attn_weights is None:
            # attn 없으면 fully-connected uniform 그래프로 가정
            adj = torch.ones(N, N, device=device)
        else:
            # head 평균 후 대칭화
            attn_mean = attn_weights.mean(dim=0)              # (N, N)
            adj = 0.5 * (attn_mean + attn_mean.transpose(0, 1))

        # self-loop 추가
        eye = torch.eye(N, device=device)
        adj = adj + eye

        # row-normalize
        row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj / row_sum                             # (N, N)

        return adj_norm

    # ----------------------------------------------------
    # 1) node-level -> graph-level logits
    # ----------------------------------------------------
    def forward(
        self,
        node_feats: Tensor,              # (N, C_in)
        attn_weights: Optional[Tensor] = None  # (H, N, N)
    ) -> Tensor:
        assert node_feats.dim() == 2
        N, C = node_feats.shape

        if N == 0:
            # 안전장치
            return node_feats.new_zeros(1, self.num_classes)

        device = node_feats.device

        # adjacency(Â) 구성
        adj = self._build_adj(attn_weights, N, device)       # (N, N)

        # -----------------------------
        # GCN layers
        # -----------------------------
        x = node_feats
        for layer in self.gcn:
            x = layer(x, adj)                                # (N, hidden)

        # -----------------------------
        # global mean pooling
        # -----------------------------
        graph_feat = x.mean(dim=0)                           # (hidden,)
        graph_feat = self.drop(graph_feat)
        logits = self.fc(graph_feat)                         # (num_classes,)

        return logits.unsqueeze(0)                           # (1, num_classes)

    # ----------------------------------------------------
    # 2) loss / predict
    # ----------------------------------------------------
    def loss(
        self,
        node_feats: Tensor,
        labels: Tensor,
        attn_weights: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        logits = self.forward(node_feats, attn_weights)
        if labels.dim() == 0:
            labels = labels.view(1)
        else:
            labels = labels.view(-1)
        loss_ce = F.cross_entropy(logits, labels)
        return dict(loss_ce=loss_ce)

    def predict(
        self,
        node_feats: Tensor,
        attn_weights: Optional[Tensor] = None
    ) -> Tensor:
        return self.forward(node_feats, attn_weights)