from collections.abc import Callable
import math

import torch
from torch.nn import functional as F


def algebraic_matmul(input: torch.Tensor, other: torch.Tensor, sum: Callable = torch.sum, mul: Callable = torch.mul) -> torch.Tensor:
    """this imlements matmul by calling add and mul. Supports everything that torch.matmul supports.

    Args:
        input (torch.Tensor): the first tensor to be multiplied
        other (torch.Tensor): the second tensor to be multiplied
        sum (Callable, optional): sum function, must support `dim` argument. Defaults to torch.sum.
        mul (Callable, optional): multiplication function. Defaults to torch.mul.

    Returns:
        torch.Tenspr: the output tensor
    """

    input_squeeze = False
    other_squeeze = False

    if input.ndim == 1:
        input_squeeze = True
        input = input.unsqueeze(0)

    if other.ndim == 1:
        other_squeeze = True
        other = other.unsqueeze(1)

    res: torch.Tensor = sum(mul(input.unsqueeze(-1), other.unsqueeze(-3)), dim = -2)

    if input_squeeze: res = res.squeeze(-2)
    if other_squeeze: res = res.squeeze(-1)

    return res

def algebraic_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    sum: Callable = torch.sum,
    mul: Callable = torch.mul,
    add: Callable = torch.add,
) -> torch.Tensor:
    """linear with custom operations

    Args:
        input (torch.Tensor): `(*, in_features)`
        weight (torch.Tensor): `(out_features, in_features)`
        bias (torch.Tensor | None, optional): `(out_features)` or `()`. Defaults to None.
        sum (Callable, optional): sum function, must support `dim` argument. Defaults to torch.sum.
        mul (Callable, optional): multiplication function. Defaults to torch.mul.
        add (Callable, optional): addition function for bias. Defaults to torch.add.

    Returns:
        torch.Tensor: `(*, out_features)`
    """
    #r = (weight @ input.unsqueeze(-1)).squeeze(-1)
    r = algebraic_matmul(weight, input.unsqueeze(-1), sum=sum, mul=mul).squeeze(-1)
    if bias is not None: return add(r, bias)
    return r



class AlgebraicLinear(torch.nn.Module):
    """linear with custom operations

    Args:
        in_channels (int): size of each input sample
        out_channels (int): size of each output sample
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        sum (Callable, optional): sum function, must support `dim` argument. Defaults to torch.sum.
        mul (Callable, optional): multiplication function. Defaults to torch.mul.
        add (Callable, optional): addition function for bias. Defaults to torch.add.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias=True,
        sum: Callable = torch.sum,
        mul: Callable = torch.mul,
        add: Callable = torch.add,
        dtype=None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, dtype=dtype))

        if bias: self.bias = torch.nn.Parameter(torch.empty(out_channels, dtype=dtype))
        else: self.bias = None

        self.sum = sum; self.mul = mul; self.add = add
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # this only makes sense for default ops but we keep it
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, x: torch.Tensor):
        return algebraic_linear(
            input = x,
            weight = self.weight,
            bias = self.bias,
            sum = self.sum,
            mul = self.mul,
            add = self.add,
        )

