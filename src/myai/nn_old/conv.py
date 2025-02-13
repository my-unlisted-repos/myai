import functools
import typing
from collections.abc import Callable, Sequence

import torch


def _get_convnd_cls(ndim: int,):
    """Returns a class."""
    if ndim == 1: return torch.nn.Conv1d
    elif ndim == 2: return torch.nn.Conv2d
    elif ndim == 3: return torch.nn.Conv3d
    else: raise ValueError(f'Invalid ndim {ndim}.')


class ConvLike(typing.Protocol):
    """Protocol for convolutional-like classes."""
    def __call__(
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


