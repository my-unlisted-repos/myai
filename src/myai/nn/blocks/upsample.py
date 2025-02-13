from collections.abc import Sequence
from typing import Any

import torch

from .block_base import Block, BlockType
from .containers import Lambda, Sequential, ensure_block


class Upsample(Block):
    def __init__(self, mode: str = "nearest", align_corners: bool | None = None):
        super().__init__()
        self.mode = mode; self.align_corners = align_corners

    def _make(self, in_channels, out_channels, channel_scale, in_size, out_size, size_scale, ndim, channel_dim):
        if channel_scale != 1: raise RuntimeError(f"Upsample can't change channels, got {channel_scale = }")
        if isinstance(size_scale, float) and size_scale <= 1: raise RuntimeError(f"Upsample can't decrease size, got {size_scale = }")

        if out_size is not None: return torch.nn.Upsample(size=tuple(out_size), mode=self.mode, align_corners=self.align_corners)
        return torch.nn.Upsample(scale_factor=size_scale, mode=self.mode, align_corners=self.align_corners)