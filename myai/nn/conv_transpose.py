import typing as T
from collections.abc import Callable, Sequence

import torch

from .act import get_act
from .dropout import get_dropout
from .norm import get_norm
from .pool import get_pool
from .upsample import get_upsample
from .containers import Sequential


def _get_convtransposend_cls(ndim: int,):
    """Returns a class."""
    if ndim == 1: return torch.nn.ConvTranspose1d
    if ndim == 2: return torch.nn.ConvTranspose2d
    if ndim == 3: return torch.nn.ConvTranspose3d
    raise ValueError(f'Invalid ndim {ndim}.')


class ConvTransposeLike(T.Protocol):
    """Protocol for transposed convolution layer like classes."""
    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        # we intentionally don't type some args because it becomes a mess
        kernel_size,
        stride: T.Any = 1,
        padding: T.Any = 0,
        output_padding: T.Any = 0,
        groups: int = 1,
        bias: bool = True,
        dilation : T.Any= 1,
        padding_mode: str = 'zeros',
        # device = None,
        dtype = None,
        ndim: int = 2,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        ...

def convtransposend(
    in_channels: int,
    out_channels: int,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    output_padding: int | Sequence[int] = 0,
    groups: int = 1,
    bias: bool = True,
    dilation : int | Sequence[int]= 1,
    padding_mode: str = 'zeros',
    # device = None,
    dtype = None,
    ndim: int = 2,
):
    kwargs = locals().copy()
    del kwargs['ndim']
    return _get_convtransposend_cls(ndim)(**kwargs)

# now this works
__test_convnd: ConvTransposeLike = convtransposend



class ConvTransposeBlock(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        output_padding: int | Sequence[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation : int | Sequence[int]= 1,
        padding_mode: str = 'zeros',
        # device = None,
        dtype = None,
        act = None,
        pool = None,
        norm = None,
        dropout = None,
        upsample = None,
        order = 'UCPAND',
        ndim: int = 2,
        conv_cls: ConvTransposeLike = convtransposend,
    ):
        modules = []
        cur_channels = in_channels

        for c in order:
            c = c.lower()

            # convolution
            if c == 'c':
                modules.append(conv_cls(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=groups,
                    bias=bias,
                    dilation=dilation,
                    padding_mode=padding_mode,
                    dtype=dtype,
                    ndim=ndim,
                ))
                cur_channels = out_channels

            # pool
            if c == 'p' and pool is not None: modules.append(get_pool(act, in_channels=cur_channels, ndim=ndim))
            # activation
            if c == 'a' and act is not None: modules.append(get_act(act, in_channels=cur_channels, ndim=ndim))
            # norm
            if c == 'n' and norm is not None: modules.append(get_norm(norm, in_channels=cur_channels, ndim=ndim))
            # dropout
            if c == 'd' and dropout is not None: modules.append(get_dropout(dropout, in_channels=cur_channels, ndim=ndim))
            # upsample
            if c == 'u' and upsample is not None: modules.append(get_upsample(upsample, in_channels=cur_channels, ndim=ndim))

        super().__init__(*modules)
