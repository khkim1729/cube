from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType


@MODELS.register_module()
class DetGraphAggregatorDetachCA(BaseModule):
    """DetGraph aggregator + detach(ref_x) + same-frame attention masking."""

    def __init__(
        self,
        in_channels: int,
        num_attention_blocks: int = 16,
        init_cfg: OptConfigType = None
    ) -> None:
        super().__init__(init_cfg)

        assert in_channels % num_attention_blocks == 0
        self.in_channels = in_channels
        self.num_attention_blocks = num_attention_blocks
        self.num_c_per_att_block = in_channels // num_attention_blocks

        # SELSA-style projections
        self.fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(
        self,
        x: Tensor,                     # (N, C)
        frame_ids: Optional[Tensor],   # (N,) 각 proposal의 프레임 id
        return_attn: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:

        roi_n, C = x.shape
        H = self.num_attention_blocks
        d = self.num_c_per_att_block

        # Q는 gradient 그대로 받고
        x_q = x
        # K/V는 detach
        ref_x = x.detach()

        # ---- Q/K embedding ----
        x_embed = self.fc_embed(x_q).view(roi_n, H, d).permute(1, 0, 2)
        ref_x_embed = self.ref_fc_embed(ref_x).view(roi_n, H, d).permute(1, 2, 0)

        # ---- attention logits ----
        attn_logits = torch.bmm(x_embed, ref_x_embed) / math.sqrt(d)   # (H, N, N)

        # ---- SAME-FRAME ATTENTION MASKING ----
        # frame_ids: (N,)
        if frame_ids is not None:
            # eq_matrix[i, j] = True if same frame
            same_frame = (frame_ids.unsqueeze(0) == frame_ids.unsqueeze(1))  # (N, N)
            # expand to (H, N, N)
            same_frame = same_frame.unsqueeze(0).expand(H, roi_n, roi_n)
            # 같은 프레임이면 softmax 전에 -inf로 마스킹
            attn_logits = attn_logits.masked_fill(same_frame, float('-inf'))

        # ---- softmax normalization ----
        attn_weights = attn_logits.softmax(dim=2)

        # ---- V projection ----
        ref_x_new = self.ref_fc(ref_x).view(roi_n, H, d).permute(1, 0, 2)

        # ---- weighted sum ----
        x_new_blocks = torch.bmm(attn_weights, ref_x_new)
        x_new = x_new_blocks.permute(1, 0, 2).contiguous().view(roi_n, C)
        x_new = self.fc(x_new)

        if return_attn:
            return x_new, attn_weights
        else:
            return x_new, None