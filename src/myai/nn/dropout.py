from collections.abc import Callable
from typing import Any
import torch

from ..python_tools import normalize_string

def _get_dropoutnd_cls(ndim: int,):
    """Returns a class."""
    if ndim == 0: return torch.nn.Dropout
    if ndim == 1: return torch.nn.Dropout1d
    if ndim == 2: return torch.nn.Dropout1d
    if ndim == 3: return torch.nn.Dropout3d
    raise ValueError(f'Invalid ndim {ndim}.')

def dropoutnd(
    p: float = 0.5,
    inplace: bool = False,
    ndim = 2,
):
    kwargs = locals().copy()
    del kwargs['ndim']
    return _get_dropoutnd_cls(ndim)(**kwargs)


def get_dropout(x, in_channels = None, ndim = 2):
    if isinstance(x, float): return dropoutnd(x, inplace=True, ndim = ndim)
    if isinstance(x, type): return x()
    if isinstance(x, Callable): return x

    raise RuntimeError(f'unknown dropout {x}')