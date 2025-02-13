import torch

from ..conv import ConvLike, convnd
from .block_base import BlockType
from .containers import Sequential
from .conv import Conv

__all__ = ["ConvBlock"]

class ConvBlock(Sequential):
    def __init__(
        self,
        kernel_size,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        dtype = None,
        act: BlockType | None = None,
        pool: BlockType | None = None,
        dropout: BlockType | None = None,
        norm: BlockType | None = None,
        order = 'CPAND',
        conv_cls: ConvLike = convnd
    ):

        factual_order = []
        blocks = []
        for char in order.lower():
            if char == 'c':
                blocks.append(Conv(kernel_size=kernel_size, groups=groups, bias = bias, padding_mode=padding_mode, dtype=dtype, conv_cls=conv_cls))
                factual_order.append(char)

            if char == 'p' and pool is not None:
                blocks.append(pool)
                factual_order.append(char)

            if char == 'a' and act is not None:
                blocks.append(act)
                factual_order.append(char)

            if char == 'n' and norm is not None:
                blocks.append(norm)
                factual_order.append(char)

            if char == 'd' and dropout is not None:
                blocks.append(dropout)
                factual_order.append(char)

        if 'p' in factual_order: size_block = 'p'
        else: size_block = 'c'
        super().__init__(blocks, channel_idx=factual_order.index('c'), size_idx=factual_order.index(size_block))