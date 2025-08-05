import typing
from collections import abc
import torch
from ..modulelist import Sequential
from ..conv import ConvLike, convnd
from ..convtranspose import ConvTransposeLike, convtransposend
from ..pool import PoolLike, maxpoolnd, make_pool, PoolType
from ..act import make_activation
from ..upsample import make_upsample

class ConvBlock(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        # we intentionally don't type some args because it becomes a mess
        kernel_size,
        stride: typing.Any = 1,
        padding: typing.Any = 0,
        dilation : typing.Any= 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        # device = None,
        dtype = None,
        ndim: int = 2,
        transposed = False,
        conv = None,
        act = None,
        norm = None,
        dropout = None,
        pool: PoolType = None,
        upsample = None,
        other = None,
        order = 'UPCANDO'
    ):
        if conv is None:
            if transposed: conv = convtransposend
            else: conv = convnd

        cur_channels = in_channels
        modules = []
        for s in order.strip().lower():
            if s == 'c':
                modules.append(conv(
                        in_channels = cur_channels,
                        out_channels = out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        padding_mode=padding_mode,
                        dtype=dtype,
                        ndim=ndim,
                    ))

                cur_channels = out_channels

            elif s == 'p': modules.append(make_pool(pool, ndim = ndim))
            elif s == 'a': modules.append(make_activation(act, in_channels = cur_channels, ndim = ndim))

            elif s == 'u': modules.append(make_upsample(upsample))