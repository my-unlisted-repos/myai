import typing as T
from collections.abc import Callable

import torch



class LinearLike(T.Protocol):
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
    def __init__(self, in_features, out_features, bias = True, dtype = None, dim = -1, module: LinearLike = Linear):
        super().__init__()
        self.linear = module(in_features = in_features, out_features = out_features, bias = bias, dtype = dtype)
        self.dim = dim

    def forward(self, input: torch.Tensor):
        if self.dim == -1: return self.linear(input)
        input = input.moveaxis(self.dim, -1)
        output = self.linear(input)
        return output.moveaxis(-1, self.dim)

__test_custom_linear: LinearLike = CustomLinear
