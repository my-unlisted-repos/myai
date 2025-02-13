from collections.abc import Callable
from typing import Any
import torch

from ..python_tools import normalize_string

def _get_batchnormnd_cls(ndim: int,):
    """Returns a class."""
    if ndim == 1: return torch.nn.BatchNorm1d
    if ndim == 2: return torch.nn.BatchNorm2d
    if ndim == 3: return torch.nn.BatchNorm3d
    raise ValueError(f'Invalid ndim {ndim}.')

def batchnormnd(
    num_features: int,
    eps: float = 0.00001,
    momentum: float | None = 0.1,
    affine: bool = True,
    track_running_stats: bool = True,
    dtype = None,
    ndim = 2,
):
    kwargs = locals().copy()
    del kwargs['ndim']
    return _get_batchnormnd_cls(ndim)(**kwargs)


def get_norm(x, in_channels = None, ndim = 2):
    if isinstance(x, type): return x()
    if isinstance(x, Callable): return x

    if x is True:
        if in_channels is None: raise ValueError("batch norm requires in channels")
        return batchnormnd(in_channels, ndim=ndim)

    if isinstance(x, str):
        x = normalize_string(x)
        if x in ('bn', 'batch', 'batchnorm'):
            if in_channels is None: raise ValueError("batch norm requires in channels")
            return batchnormnd(in_channels, ndim=ndim)

        if x == 'fixedbn':
            if in_channels is None: raise ValueError("batch norm requires in channels")
            return batchnormnd(in_channels, ndim=ndim, track_running_stats=False)

    raise RuntimeError(f'unknown norm {x}')