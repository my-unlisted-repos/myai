from collections.abc import Callable

import torch

from ..python_tools import normalize_string


def get_act(x, in_channels = None, ndim = 2):
    if isinstance(x, type): return x()
    if isinstance(x, Callable): return x

    if isinstance(x, str):
        x = normalize_string(x)
        if x == 'relu': return torch.nn.ReLU(True)
        if x == 'gelu': m = torch.nn.GELU()
        if x == 'sigmoid': m = torch.nn.Sigmoid()
        if x == 'softmax': m = torch.nn.Softmax()

    raise RuntimeError(f'unknown pool {x}')