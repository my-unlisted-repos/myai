import typing as T
from collections.abc import Callable, Sequence

import torch


def _get_convtransposend_cls(ndim: int,):
    """Returns a class."""
    if ndim == 1: return torch.nn.ConvTranspose1d
    elif ndim == 2: return torch.nn.ConvTranspose2d
    elif ndim == 3: return torch.nn.ConvTranspose3d
    else: raise ValueError(f'Invalid ndim {ndim}.')


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
