from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType


@MODELS.register_module()
class DetGraphAggregator(BaseModule):

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        init_cfg: OptConfigType = None
    ) -> None:
        super().__init__(init_cfg)
        assert in_channels % num_heads == 0, \
            f'in_channels({in_channels}) must be divisible by num_heads({num_heads})'

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

    def _reshape_to_heads(self, x: Tensor) -> Tensor:
        """(N, C) -> (H, N, C/H)"""
        N, C = x.shape
        x = x.view(N, self.num_heads, self.head_dim)   # (N, H, d)
        x = x.permute(1, 0, 2).contiguous()            # (H, N, d)
        return x

    def forward(
        self,
        x: Tensor,
        frame_ids: Optional[Tensor] = None,
        return_attn: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        N, C = x.shape
        assert C == self.in_channels

        q = self.q_proj(x)   # (N, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._reshape_to_heads(q)             # (H, N, d)
        k = self._reshape_to_heads(k).transpose(1, 2)  # (H, d, N)
        v = self._reshape_to_heads(v)             # (H, N, d)

        attn_logits = torch.bmm(q, k) / math.sqrt(self.head_dim)  # (H, N, N)
        attn_weights = attn_logits.softmax(dim=-1)                 # (H, N, N)

        out_heads = torch.bmm(attn_weights, v)                     # (H, N, d)
        out = out_heads.permute(1, 0, 2).contiguous().view(N, C)   # (N, C)
        x_new = self.out_proj(out)                                 # (N, C)

        if return_attn:
            return x_new, attn_weights
        else:
            return x_new, None
