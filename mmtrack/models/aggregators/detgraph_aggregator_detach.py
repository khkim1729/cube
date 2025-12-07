from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType


@MODELS.register_module()
class DetGraphAggregatorDetach(BaseModule):
    """SELSA-style self-attention aggregator for DetGraph.

    - 구조는 SELSA의 SelsaAggregator를 최대한 그대로 사용
      (fc_embed, ref_fc_embed, ref_fc, fc, num_attention_blocks 등)
    - 차이점: ref_x를 따로 받지 않고, x 자체를 reference로 사용하는
      self-attention 형태로 동작
    - ★ 추가 차이: ref_x = x.detach() 로 두어
      각 proposal의 loss가 다른 proposal feature로는 gradient를 보내지 않음
      (forward 값은 그대로, backward 경로만 SELSA스럽게 만듦)
    - 출력: (N, C) aggregated features, (H, N, N) attention weights
    """

    def __init__(
        self,
        in_channels: int,
        num_attention_blocks: int = 16,
        init_cfg: OptConfigType = None
    ) -> None:
        super().__init__(init_cfg)

        assert in_channels % num_attention_blocks == 0, \
            f'in_channels({in_channels}) must be divisible by ' \
            f'num_attention_blocks({num_attention_blocks})'

        self.in_channels = in_channels
        self.num_attention_blocks = num_attention_blocks
        self.num_c_per_att_block = in_channels // num_attention_blocks

        # SELSA-style projection layers
        self.fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(
        self,
        x: Tensor,
        frame_ids: Optional[Tensor] = None,
        return_attn: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Self-attention version of SELSA aggregator.

        Args:
            x (Tensor): (N, C) proposal features.
            frame_ids (Tensor, optional): (N,) frame index (현재는 사용하지 않음).
            return_attn (bool): True일 때 (H, N, N) attention map 반환.

        Returns:
            x_new (Tensor): (N, C) aggregated features.
            attn_weights (Tensor or None): (H, N, N) attention weights.
        """
        roi_n, C = x.shape
        assert C == self.in_channels

        H = self.num_attention_blocks
        d = self.num_c_per_att_block

        # -----------------------------
        # 1) Query / Key embedding (SELSA 수식 그대로)
        # -----------------------------
        # self-attention: query는 x에서, key/value는 ref_x에서
        # ★ ref_x는 detach해서 다른 proposal 쪽으로 gradient가 안 흐르게 함
        ref_x = x.detach()          # <- 여기만 기존 ref_x = x 에서 변경

        # Q from x
        x_embed = self.fc_embed(x)              # (N, C)
        x_embed = x_embed.view(roi_n, H, d)     # (N, H, d)
        x_embed = x_embed.permute(1, 0, 2)      # (H, N, d)

        # K from ref_x (detached)
        ref_x_embed = self.ref_fc_embed(ref_x)      # (N, C)
        ref_x_embed = ref_x_embed.view(roi_n, H, d) # (N, H, d)
        ref_x_embed = ref_x_embed.permute(1, 2, 0)  # (H, d, N)

        # -----------------------------
        # 2) Attention weights (H, N, N)
        #    weights = softmax(Q K^T / sqrt(d))
        # -----------------------------
        attn_logits = torch.bmm(x_embed, ref_x_embed) / math.sqrt(d)  # (H, N, N)
        attn_weights = attn_logits.softmax(dim=2)                     # (H, N, N)

        # -----------------------------
        # 3) Value projection + weighted sum
        # -----------------------------
        ref_x_new = self.ref_fc(ref_x)            # (N, C)
        ref_x_new = ref_x_new.view(roi_n, H, d)   # (N, H, d)
        ref_x_new = ref_x_new.permute(1, 0, 2)    # (H, N, d)

        # (H, N, N) @ (H, N, d) → (H, N, d)
        x_new_blocks = torch.bmm(attn_weights, ref_x_new)  # (H, N, d)

        # (H, N, d) → (N, H, d) → (N, C)
        x_new = x_new_blocks.permute(1, 0, 2).contiguous() # (N, H, d)
        x_new = x_new.view(roi_n, C)                       # (N, C)

        # 최종 projection (SELSA와 동일)
        x_new = self.fc(x_new)                             # (N, C)

        if return_attn:
            return x_new, attn_weights
        else:
            return x_new, None