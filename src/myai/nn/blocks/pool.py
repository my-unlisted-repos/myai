from collections.abc import Sequence
from typing import Any

from ..pool import maxpoolnd, avgpoolnd
from .block_base import Block, BlockType
from .containers import Sequential, Lambda, ensure_block


class MaxPool(Block):
    def _make(self, in_channels, out_channels, channel_scale, in_size, out_size, size_scale, ndim, channel_dim):
        if channel_scale != 1: raise RuntimeError(f"MaxPool can't change channels, got {channel_scale = }")
        return maxpoolnd(size_scale, ndim = ndim)

class AvgPool(Block):
    def _make(self, in_channels, out_channels, channel_scale, in_size, out_size, size_scale, ndim, channel_dim):
        if channel_scale != 1: raise RuntimeError(f"AvgPool can't change channels, got {channel_scale = }")
        return avgpoolnd(size_scale, ndim = ndim)

