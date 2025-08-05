import typing
from collections.abc import Sequence, Callable

import torch

class PoolLike(typing.Protocol):
    """Protocol for pooling classes."""
    def __call__(
        self,
        kernel_size,
        stride: typing.Any = None,
        padding: typing.Any = 0,
        dilation: typing.Any = 1,
        ndim: int = 2,
    ) -> Callable[[torch.Tensor], torch.Tensor]: ...

def _get_maxpoolnd_cls(ndim: int,):
    """Returns a class."""
    if ndim == 1: return torch.nn.MaxPool1d
    elif ndim == 2: return torch.nn.MaxPool2d
    elif ndim == 3: return torch.nn.MaxPool3d
    else: raise ValueError(f'Invalid ndim {ndim}.')

def maxpoolnd(
    kernel_size,
    stride: typing.Any = None,
    padding: typing.Any = 0,
    dilation: typing.Any = 1,
    ndim = 2,
):
    kwargs = locals().copy()
    del kwargs['ndim']
    return _get_maxpoolnd_cls(ndim)(**kwargs)

__test_maxpoolnd: PoolLike = maxpoolnd

def _maybe_map(x, fn):
    if isinstance(x, Sequence): return [fn(y) for y in x]
    return fn(x)

PoolType = PoolLike | Callable[..., PoolLike] | str | None | int | Sequence[int] | Sequence[str | int] | dict
def make_pool(
    m: PoolType,
    ndim = 2,
) -> None | Callable[[torch.Tensor], torch.Tensor]:
    if m is None: return  None

    if isinstance(m, str) and ',' in m: m = [i.lower().strip() for i in m.split(',')]
    module = name = kernel_size = stride = padding = dilation = None

    if isinstance(m, torch.nn.Module): return m
    if isinstance(m, type): return m(ndim)
    if isinstance(m, int): return maxpoolnd(m, m, ndim = ndim)
    if isinstance(m, Sequence) and (not isinstance(m, str)):

        if not isinstance(m[0], (int, Sequence)):
            _x = m[0]
            if isinstance(_x, str): name = _x
            else: module = _x
            m = m[1:]

        if len(m) >= 1: kernel_size = _maybe_map(m[0], int)
        if len(m) >= 2: stride = _maybe_map(m[1], int)
        if len(m) >= 3: padding = _maybe_map(m[2], int)
        if len(m) >= 4: dilation = _maybe_map(m[3], int)

    if isinstance(m, str): name = m
    if isinstance(name, str): name = name.strip().lower()

    if module is None:
        if name is not None:
            if name in ('max', 'maxpool'): module = maxpoolnd
            else: raise NotImplementedError(f'{name = }')
        else:
            module = maxpoolnd

    return module(kernel_size, stride, padding, dilation, ndim = ndim)