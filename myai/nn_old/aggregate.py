import collections.abc as A
import torch
from .modulelist import ModuleList
from .linear import LinearLike, Linear
from .conv import ConvLike, convnd
from .convtranspose import ConvTransposeLike, convtransposend

__all__ = [
    "maxout", "minout", "minmaxsum", "maxmindiff", "maxmindist", "minmaxprod", "irfftout",
    "Aggregate", "AggLinear", "AggConv", "AggConvTranspose",
]
def maxout(x:torch.Tensor):
    return torch.amax(x, dim = 0)

def minout(x: torch.Tensor):
    return torch.amin(x, dim = 0)

def minmaxsum(x: torch.Tensor):
    return maxout(x) + minout(x)

def maxmindiff(x: torch.Tensor):
    return maxout(x) - minout(x)

def maxmindist(x: torch.Tensor):
    return maxmindiff(x).abs()

def minmaxprod(x: torch.Tensor):
    return maxout(x) * minout(x)

def irfftout(x: torch.Tensor):
    if x.size(0) != 2: raise ValueError(f'irfftout only works with 2 modules, got x of size {x.size()}')
    return torch.fft.irfft(torch.complex(*x), s = x.shape[2:]) # pylint:disable = E1102

class Aggregate(ModuleList):
    """Aggregates the output of multiple layers."""
    def __init__(
        self,
        modules: list[torch.nn.Module | A.Callable],
        agg_fn: torch.nn.Module | A.Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__(modules)
        self.agg_fn = agg_fn

    def forward(self, x: torch.Tensor):
        return self.agg_fn(torch.stack([m(x) for m in self], dim=0))

class AggLinear(Aggregate):
    """Multiple linear layers where output is element-wise maximum of inputs or some other aggregation function."""
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=None,
        dim=-1,
        module: LinearLike = Linear,
        num=2,
        agg_fn = maxout,
    ):
        super().__init__(
            [module(
                in_features = in_features,
                out_features = out_features,
                bias = bias,
                dtype = dtype,
                dim = dim
            ) for _ in range(num)], agg_fn = agg_fn)

class AggConv(Aggregate):
    """Multiple conv layers where output is element-wise maximum of inputs or some other aggregation function."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | A.Sequence[int],
        stride: int | A.Sequence[int] = 1,
        padding: int | A.Sequence[int] = 0,
        dilation: int | A.Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        dtype = None,
        ndim = 2,
        module: ConvLike = convnd,
        num: int = 2,
        agg_fn = maxout,
    ):
        kwargs = locals().copy()
        module = kwargs.pop('module')
        num = kwargs.pop('num')
        agg_fn = kwargs.pop('agg_fn')
        del kwargs['self'], kwargs['__class__']

        super().__init__([module(**kwargs) for _ in range(num)], agg_fn = agg_fn)

__aggconv_test: ConvLike = AggConv

class AggConvTranspose(Aggregate):
    """Multiple conv layers where output is element-wise maximum of inputs or some other aggregation function."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | A.Sequence[int],
        stride: int | A.Sequence[int] = 1,
        padding: int | A.Sequence[int] = 0,
        output_padding: int | A.Sequence[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation : int | A.Sequence[int]= 1,
        padding_mode: str = 'zeros',
        # device = None,
        dtype = None,
        ndim: int = 2,
        module: ConvTransposeLike = convtransposend,
        num: int = 2,
        agg_fn = maxout,
    ):
        kwargs = locals().copy()
        module = kwargs.pop('module')
        num = kwargs.pop('num')
        agg_fn = kwargs.pop('agg_fn')
        del kwargs['self'], kwargs['__class__']

        super().__init__([module(**kwargs) for _ in range(num)], agg_fn = agg_fn)

__aggconvtranspose_test: ConvTransposeLike = AggConvTranspose
