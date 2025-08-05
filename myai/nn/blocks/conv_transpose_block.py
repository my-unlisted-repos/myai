import torch

from .block_base import BlockType
from .containers import Sequential
from .conv_transpose import ConvTranspose, ConvTransposeLike, convtransposend
from .pool import AvgPool, MaxPool

__all__ = ["ConvTransposeBlock"]

class ConvTransposeBlock(Sequential):
    def __init__(
        self,
        kernel_size,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        dtype = None,
        act: BlockType | None = None,
        upsample: BlockType | None = None,
        dropout: BlockType | None = None,
        norm: BlockType | None = None,
        order = 'UPCAND',
        conv_cls: ConvTransposeLike = convtransposend
    ):

        factual_order = []
        blocks = []
        for char in order.lower():
            if char == 'c':
                blocks.append(ConvTranspose(kernel_size=kernel_size, groups=groups, bias = bias, padding_mode=padding_mode, dtype=dtype, conv_cls=conv_cls))
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

            if char == 'u' and upsample is not None:
                blocks.append(upsample)
                factual_order.append(char)

        if 'u' in factual_order: size_block = 'u'
        else: size_block = 'c'
        super().__init__(blocks, channel_idx=factual_order.index('c'), size_idx=factual_order.index(size_block))