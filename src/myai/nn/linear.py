import functools
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import torch

from .act import get_act
from .containers import Sequential
from .dropout import get_dropout
from .norm import get_norm
from .pool import get_pool, maxpoolnd


class LinearLike(Protocol):
    """Protocol for linear-like classes, anything that changes size of one dimension."""
    def __call__(
        self, in_features: int, out_features: int, bias: bool = True, dtype=None, dim=-1
    ) -> Callable[[torch.Tensor], torch.Tensor]: ...


class Linear(torch.nn.Linear):
    """A linear layer but can be applied among any axis."""
    def __init__(self, in_features, out_features, bias = True, dtype = None, dim = -1,):
        super().__init__(in_features = in_features, out_features = out_features, bias = bias, dtype = dtype)
        self.dim = dim

    def forward(self, input: torch.Tensor):
        if self.dim == -1: return super().forward(input)
        input = input.moveaxis(self.dim, -1)
        output = super().forward(input)
        return output.moveaxis(-1, self.dim)

__test_linear: LinearLike = Linear

class CustomLinear(torch.nn.Module):
    """A linear layer but can be applied among any axis, and you can choose any similar module."""
    def __init__(self, in_features, out_features, bias = True, dtype = None, dim = -1, module = Linear):
        super().__init__()
        self.linear = module(in_features = in_features, out_features = out_features, bias = bias, dtype = dtype)
        self.dim = dim

    def forward(self, input: torch.Tensor):
        if self.dim == -1: return self.linear(input)
        input = input.moveaxis(self.dim, -1)
        output = self.linear(input)
        return output.moveaxis(-1, self.dim)

__test_custom_linear: LinearLike = CustomLinear


class LinearBlock(Sequential):
    def __init__(
        self,
        in_features,
        out_features,
        bias = True,
        dtype = None,
        dim = -1,
        act = None,
        dropout = None,
        flatten = False,
        order = 'FLAD',
        ndim = 0,
        linear_cls: LinearLike = Linear
    ):
        modules = []
        cur_channels = in_features

        for char in order.lower():
            # convolution
            if char == 'l':
                modules.append(linear_cls(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias,
                    dtype=dtype,
                    dim=dim,
                ))
                cur_channels = out_features

            # activation
            if char == 'a' and act is not None: modules.append(get_act(act, in_channels=cur_channels, ndim=ndim))
            # dropout
            if char == 'd' and dropout is not None: modules.append(get_dropout(dropout, in_channels=cur_channels, ndim=ndim))
            # flatten
            if char == 'f' and flatten: modules.append(torch.nn.Flatten())

        super().__init__(*modules)
