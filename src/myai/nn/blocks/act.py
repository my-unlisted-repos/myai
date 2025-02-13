from collections.abc import Sequence
from typing import Any

from ..act import get_act
from .block_base import Block, BlockType
from .containers import Sequential, Lambda, ensure_block


class Act(Block):
    def __init__(self, type: Any = 'relu'):
        super().__init__()
        self.type = type

    def _make(self, in_channels, out_channels, channel_scale, in_size, out_size, size_scale, ndim, channel_dim):
        if channel_scale != 1: raise RuntimeError(f"Act can't change channels, got {channel_scale = }")
        if size_scale != 1: raise RuntimeError(f"Act can't change size, got {size_scale = }")

        return get_act(self.type, in_channels=in_channels, ndim = ndim)