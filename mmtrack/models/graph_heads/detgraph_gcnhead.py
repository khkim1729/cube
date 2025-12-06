from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.model import BaseModule

from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType

# ★ PyTorch Geometric
from torch_geometric.nn import GCNConv


@MODELS.register_module()
class DetGraphGCNHead(BaseModule):
    """PyG GCNConv + global pooling 기반 video-level graph head.

    Inputs:
        - node_feats: (N_nodes, C)
        - attn_weights (optional): (H, N_nodes, N_nodes)

    동작:
        1) attn_weights → soft adjacency (N,N) → edge_index (2,E), edge_weight (E,)
        2) GCNConv stack 돌림
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
        # GCN stack (PyG GCNConv)
        # -----------------------------
        gcn_list = []
        in_c = in_channels
        for _ in range(gcn_layers):
            gcn_list.append(GCNConv(in_c, hidden_channels))
            in_c = hidden_channels
        self.gcn = nn.ModuleList(gcn_list)

        # -----------------------------
        # Readout + classifier
        # -----------------------------
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.drop = nn.Dropout(dropout)

    # ----------------------------------------------------
    # attn_weights → edge_index, edge_weight
    # ----------------------------------------------------
    def _build_edges(
        self,
        attn_weights: Optional[Tensor],
        N: int,
        device: torch.device
    ) -> (Tensor, Optional[Tensor]):
        """attn_weights(H,N,N) 를 PyG용 edge_index, edge_weight로 변환.

        edge_index: (2, E)
        edge_weight: (E,) or None
        """
        if attn_weights is None:
            # attn 없으면 fully-connected uniform 그래프
            adj = torch.ones(N, N, device=device)
        else:
            # head 평균 후 대칭화 (i->j, j->i 모두 반영)
            attn_mean = attn_weights.mean(dim=0)              # (N, N)
            adj = 0.5 * (attn_mean + attn_mean.transpose(0, 1))

        # self-loop 추가 (GCNConv도 self-loop를 추가할 수 있지만,
        # 여기선 확실히 포함해 둠)
        eye = torch.eye(N, device=device)
        adj = adj + eye

        # 완전 dense 그래프 → 모든 nonzero edge 사용
        # (CEUS에서 N ≤ 16 정도라 N^2 edge도 부담 없음)
        edge_index = adj.nonzero(as_tuple=False).t()          # (2, E)
        edge_weight = adj[edge_index[0], edge_index[1]]       # (E,)

        return edge_index.long(), edge_weight

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

        # PyG용 edge_index, edge_weight 구성
        edge_index, edge_weight = self._build_edges(attn_weights, N, device)

        # -----------------------------
        # GCNConv layers
        # -----------------------------
        x = node_feats
        for conv in self.gcn:
            x = conv(x, edge_index, edge_weight)   # (N, hidden)
            x = F.relu(x)

        # -----------------------------
        # global mean pooling
        # -----------------------------
        graph_feat = x.mean(dim=0)                 # (hidden,)
        graph_feat = self.drop(graph_feat)
        logits = self.fc(graph_feat)               # (num_classes,)

        return logits.unsqueeze(0)                 # (1, num_classes)

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