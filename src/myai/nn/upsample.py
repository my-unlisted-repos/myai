from collections.abc import Callable
from typing import Any
import torch

from ..python_tools import normalize_string


def get_upsample(x, in_channels = None, ndim = 2):
    if isinstance(x, type): return x()
    if isinstance(x, Callable): return x

    if isinstance(x, (int, float)):
        return torch.nn.Upsample(scale_factor = x, mode = 'bilinear')

    raise RuntimeError(f'unknown upsample {x}')