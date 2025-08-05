from typing import Literal, Any, Protocol
import functools
from collections.abc import Callable, Sequence

import torch

from .act import get_act
from .containers import Sequential
from .dropout import get_dropout
from .norm import get_norm
from .pool import get_pool, maxpoolnd
from .upsample import get_upsample

def _get_convnd_cls(ndim: int,):
    """Returns a class."""
    if ndim == 1: return torch.nn.Conv1d
    if ndim == 2: return torch.nn.Conv2d
    if ndim == 3: return torch.nn.Conv3d
    raise ValueError(f'Invalid ndim {ndim}.')


class ConvLike(Protocol):
    """Protocol for convolutional-like classes."""
    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        # we intentionally don't type some args because it becomes a mess
        kernel_size,
        stride: Any = 1,
        padding: Any = 0,
        dilation : Any= 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        # device = None,
        dtype = None,
        ndim: int = 2,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        ...


def convnd(
    in_channels: int,
    out_channels: int,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = 'zeros',
    dtype = None,
    ndim = 2,
):
    kwargs = locals().copy()
    del kwargs['ndim']
    return _get_convnd_cls(ndim)(**kwargs)

# now this works
__test_convnd: ConvLike = convnd


class ConvBlock(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Literal['same'] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        dtype = None,
        act = None,
        norm = None,
        pool = None,
        dropout = None,
        upsample = None,
        order = 'UCPAND',
        ndim = 2,
        conv_cls: ConvLike = convnd
    ):
        modules = []
        cur_channels = in_channels

        for char in order:
            char = char.lower()

            # convolution
            if char == 'c':
                modules.append(conv_cls(
                    in_channels=in_channels,
                    out_channels=out_channels,
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

            # pool
            if char == 'p' and pool is not None: modules.append(get_pool(act, in_channels=cur_channels, ndim=ndim))
            # activation
            if char == 'a' and act is not None: modules.append(get_act(act, in_channels=cur_channels, ndim=ndim))
            # norm
            if char == 'n' and norm is not None: modules.append(get_norm(norm, in_channels=cur_channels, ndim=ndim))
            # dropout
            if char == 'd' and dropout is not None: modules.append(get_dropout(dropout, in_channels=cur_channels, ndim=ndim))
            # upsample
            if char == 'u' and upsample is not None: modules.append(get_upsample(upsample, in_channels=cur_channels, ndim=ndim))


        super().__init__(*modules)
