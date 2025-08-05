from collections.abc import Sequence
from typing import Any

from ..dropout import get_dropout
from .block_base import Block, BlockType
from .containers import Sequential, Lambda, ensure_block


class Dropout(Block):
    def __init__(self, p: Any = 0.5):
        super().__init__()
        self.p = p

    def _make(self, in_channels, out_channels, channel_scale, in_size, out_size, size_scale, ndim, channel_dim):
        if channel_scale != 1: raise RuntimeError(f"Dropout can't change channels, got {channel_scale = }")
        if size_scale != 1: raise RuntimeError(f"Dropout can't change size, got {size_scale = }")

        return get_dropout(self.p, in_channels=in_channels, ndim = ndim)