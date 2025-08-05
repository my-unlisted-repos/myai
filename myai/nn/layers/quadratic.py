import math
from functools import cache

import torch
from torch.nn import functional as F


@cache
def tri_numel(N):
    """returns number of element in triangle of N x N matrix"""
    return torch.triu_indices(N,N).size(-1)

def tril_flatten(input):
    """returns lower triangle flattened, supports batches"""
    N = input.size(-1)
    indicies = torch.tril_indices(N, N)
    indicies = N * indicies[0] + indicies[1]
    return input.flatten(-2)[..., indicies]

def quadratic_monomials(x: torch.Tensor):
    """you pass `[a,b,c]` it returs `[a^2, ab, ac, b^2, bc, c^2]`"""
    return tril_flatten((x.unsqueeze(-1) @ x.unsqueeze(-2)))

def quadratic(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None = None, linear_terms=True):
    """quadratic layer like linear but more inputs.

    Args:
        x (torch.Tensor): input `(*, num_features)`
        W (torch.Tensor): weight `(out_features, [num tri elements in in_features] + in_features)`
        b (torch.Tensor | None): `(out_features)` or `()`
    """
    x_quad = quadratic_monomials(x)

    if linear_terms:
        x_quad = torch.cat((x_quad, x), -1)

    # apply linear to it
    ret = F.linear(x_quad, W, b) # pylint:disable=not-callable

    return ret

class Quadratic(torch.nn.Module):
    """essentially a polynomial regiression layer, also note that this is a non linear tansformation and composing
    this leads to higher order transformations and activations dont seem to work well too

    Args:
        in_channels (int): size of each input sample
        out_channels (int): size of each output sample
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        linear_terms (bool, optional): if True, uses both linear and quadratic terms. Defaults to True.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, linear_terms: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_channels, tri_numel(in_channels) + in_channels*linear_terms))
        if bias: self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else: self.bias = None
        self.linear_terms = linear_terms
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # maybe i can figure out better init
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return quadratic(x, self.weight, self.bias, self.linear_terms)