from collections.abc import Sequence
from typing import Any

from ..linear import LinearLike, Linear as _Linear
from .block_base import Block, BlockType

__all__ = ["Linear"]

class Linear(Block):
    def __init__(self, bias = True, dtype = None, linear_cls: LinearLike = _Linear):
        super().__init__()
        self.bias = bias
        self.dtype = dtype
        self.linear_cls = linear_cls

    def _make(self, in_channels, out_channels, channel_scale, in_size, out_size, size_scale, ndim, channel_dim):
        if size_scale != 1: raise ValueError(f"Linear can't change size, got {locals() = }")
        if in_channels is None: raise RuntimeError("Linear requires in_channels")
        if out_channels is None: raise RuntimeError("Linear requires out_channels")

        return self.linear_cls(
            in_features=in_channels,
            out_features=out_channels,
            bias = self.bias,
            dtype = self.dtype,
            dim = channel_dim,
        )