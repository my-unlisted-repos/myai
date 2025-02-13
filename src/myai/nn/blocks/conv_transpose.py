from collections.abc import Sequence
from typing import Any

from ..conv_transpose import ConvTransposeLike, convtransposend
from .block_base import Block, BlockType

__all__ = ["ConvTranspose"]

class ConvTranspose(Block):
    def __init__(
        self,
        kernel_size,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        dtype = None,
        conv_cls: ConvTransposeLike = convtransposend
    ):
        super().__init__()
        kwargs = locals().copy()
        del kwargs['self'], kwargs['__class__']

        self.kernel_size = kwargs.pop('kernel_size')
        self.conv_cls = kwargs.pop('conv_cls')

        self.kwargs = kwargs

    def _make(self, in_channels, out_channels, channel_scale, in_size, out_size, size_scale, ndim, channel_dim):
        if in_channels is None: raise RuntimeError("Conv requires in_channels")
        if out_channels is None: raise RuntimeError("Conv requires out_channels")
        if isinstance(size_scale, Sequence): raise NotImplementedError(f"Conv doesn't support size_scale tuple, got {size_scale = }")

        stride = size_scale
        if stride % 1 != 0: raise RuntimeError(f"size_scale must be an integer, got {size_scale = }")
        stride = int(stride)

        # supported cases
        # kernel_size = stride
        if self.kernel_size == stride:
            padding = 0; output_padding = 0

        # kernel_size is 3 and stride is 2
        elif self.kernel_size == 3 and stride == 2:
            padding = 1; output_padding = 1

        else: raise NotImplementedError(f"{self.kernel_size = }, {stride = }")

        return self.conv_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding,
            ndim = ndim,
            **self.kwargs
        )
