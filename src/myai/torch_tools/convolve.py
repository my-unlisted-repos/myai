from collections.abc import Sequence
from typing import Literal
import torch
from .pad_ import pad
from .crop_ import crop_to_shape

def _get_conv_patches(
    in1: torch.Tensor,
    in2: torch.Tensor,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    mode: Literal['full', 'valid', 'same'] = "valid",
    padding_mode="constant",
    value=None,
):
    """Returns (*in.shape, *patch.shape), and patch.ndim

    Args:
        in1 (torch.Tensor): _description_
        in2 (torch.Tensor): _description_
        stride (int | Sequence[int], optional): _description_. Defaults to 1.
        mode (Literal[&#39;full&#39;, &#39;valid&#39;, &#39;same&#39;], optional): _description_. Defaults to "valid".
        padding (int | Sequence[int], optional): _description_. Defaults to 0.
        padding_mode (str, optional): _description_. Defaults to "constant".
        value (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # pytorch style broadcasting,
    # e.g if you have a (16×3×128×128) input and (3×3) kernel,
    # pytorch will broadcast the kernel to (1×1×3×3).
    in2_shape = list(in2.size())
    if len(in2_shape) < in1.dim():
        in2_shape = [1] * (in1.dim() - len(in2_shape)) + in2_shape

    # make stride into a per-dimension list
    if isinstance(stride, int): stride = [stride] * in1.dim()

    # add padding if mode is `full` or `same`
    if mode in ('full', 'same'):
        if isinstance(padding, int): padding = [padding] * in1.dim()
        padding = [p + 2 * (s - 1) for p, s in zip(padding, in2_shape)]

    # apply padding
    if padding != 0:
        if isinstance(padding, int): padding = [padding] * in1.dim()
        in1 = pad(in1, padding, mode=padding_mode, value=value, where='center')

    # number of dimensions to sum over
    nreducedims = 0
    # unfold per each dimension
    for i, dimsize in enumerate(in2_shape):
        # only unfold if kernel size or stride is greater than 1
        if dimsize > 1 or stride[i] > 1:
            # print(in1.shape, i, dimsize, stride[i])
            in1 = in1.unfold(i, dimsize, stride[i])
        else:
            in1 = in1.unsqueeze(-1)
        nreducedims += 1

    return in1, nreducedims

def correlate(
    in1: torch.Tensor,
    in2: torch.Tensor,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    mode: Literal['full', 'valid', 'same'] = "valid",
    padding_mode="constant",
    value=None,
) -> torch.Tensor:
    """Cross-correlate two N-dimensional arrays.

    This should be identical to `scipy.signal.correlate`, but for pytorch and differentiable.
    Note that this probably isn't as performant as torch.nn.functional.convnd.

    Args:
        in1 (torch.Tensor): First input.
        in2 (torch.Tensor): Second input. Should have the same number of dimensions as in1 or less (will be broadcasted).
        stride (int | Sequence[int], optional): Convolution stride. Defaults to 1.
        mode (str): Convolution mode.
        padding (int | Sequence[int], optional): Amount of padding to `in1` before convolution, per dimension. Defaults to 0.
        padding_mode (str, optional): Padding mode. Defaults to 'constant'.
        value (_type_, optional): Padding constant value. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    patches, nreducedims = _get_conv_patches(
        in1,
        in2,
        stride=stride,
        mode=mode,
        padding=padding,
        padding_mode=padding_mode,
        value=value,
    )

    # if in1 is (3×128×128) and  in2 is (1×4×4), we get (3×126×126×4×4) after unfolding.
    # we multiply that by our (4×4) kernel, which means each patch is multiplied by that,
    # `nsumdims` is 2, so we sum over those last two dimensions.
    if mode in ('valid', 'full'): return (patches * in2).sum([-i for i in range(1, nreducedims + 1)])
    # same mode
    return crop_to_shape((patches * in2).sum([-i for i in range(1, nreducedims + 1)]), shape = in1.size(), where='center')

def convolve(
    in1: torch.Tensor,
    in2: torch.Tensor,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    mode: Literal['full', 'valid', 'same'] = "valid",
    padding_mode="constant",
    value=None,
    ) -> torch.Tensor:
    """Convolve two N-dimensional arrays.

    This should be identical to `scipy.signal.convolve`, but for pytorch and differentiable.
    Note that this probably isn't as performant as torch.nn.functional.convnd.

    Args:
        in1 (torch.Tensor): First input.
        in2 (torch.Tensor): Second input. Should have the same number of dimensions as in1 or less (will be broadcasted).
        stride (int | Sequence[int], optional): Convolution stride. Defaults to 1.
        mode (str): Convolution mode.
        padding (int | Sequence[int], optional): Amount of padding to `in1` before convolution, per dimension. Defaults to 0.
        padding_mode (str, optional): Padding mode. Defaults to 'constant'.
        value (_type_, optional): Padding constant value. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    return correlate(
        in1=in1,
        in2=torch.flip(in2, list(range(in2.dim()))),
        stride=stride,
        mode=mode,
        padding=padding,
        padding_mode=padding_mode,
        value=value,
    )


def _prepare_stride_or_padding(
    input_dim,
    params: int | Sequence[int],
    expand_with: int,
):
    if isinstance(params, int): params = [expand_with] * 3 + [params] * (input_dim - 3)
    elif len(params) < input_dim: params = [expand_with] * (input_dim - len(params)) + list(params)
    if any([i != expand_with for i in params[:3]]): raise ValueError(f"First 3 dimensions have to be {expand_with}, got {params}!") # type:ignore
    return params

def conv_layer(
    input: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    padding_mode = "constant",
    value = None,
):
    """An implementation of torch.nn.functional.conv(n)d, that can be experimented with,
    e.g. you can calculate absolute difference to kernel instead of multiplying by it, or anything else.
    That being said it is extremely slow on CPU, but seems to be okay on GPU, but is still likely way slower than F.conv2d

    Args:
        input (torch.Tensor): Input tensor of shape (batch, in_channels, *spatial).
        kernel (torch.Tensor): Kernel tensor of shape (out_channels, in_channels, *kernel_size).
        bias (torch.Tensor, optional): Optional bias of shape (out_channels). Defaults to None.
        stride (int | Sequence[int], optional): Stride. Defaults to 1.
        padding (int | Sequence[int], optional): Padding (this differs from pytorch, so with padding = 1 it will add 1 whereas pytorch adds 2). Defaults to 0.
        mode (Literal[&#39;full&#39;, &#39;valid&#39;, &#39;same&#39;], optional): Convolution mode. Defaults to "valid".
        padding_mode (str, optional): _description_. Defaults to "constant".
        value (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    kernel = kernel.unsqueeze(1)

    input_dim = input.dim() + 1
    stride = _prepare_stride_or_padding(input_dim, stride, 1)
    padding = _prepare_stride_or_padding(input_dim, padding, 0)

    patches, nreducedims = _get_conv_patches(
        input.expand(kernel.size(0), *input.size()),
        kernel,
        stride = stride,
        mode = 'valid',
        padding = padding,
        padding_mode = padding_mode,
        value = value,
    )

    # this is the part that can be swapped to get a different operation similar to convolution.
    # `patches` is (1, batch_size, 1, *spatial_size, out_channels, 1, in_channels, *kernel_size)
    # `K` is (out_channels, 1, in_channels, *kernel_size)
    # so `K` is broadcastable to `patches`.
    if bias is None: return (patches.squeeze(2).squeeze(0) * kernel).sum([-i for i in range(1, nreducedims)]).movedim(-1, 1)
    else: return ((patches.squeeze(2).squeeze(0) * kernel).sum([-i for i in range(1, nreducedims)]) + bias).movedim(-1, 1)

class CustomConvnd(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size:int | Sequence[int] ,
        stride:int | Sequence[int]  = 1,
        padding:int | Sequence[int] = 0,
        dilation: Literal[1] = 1,
        groups: Literal[1] = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None,
        ndim=2,
    ):
        """A module that copies torch.nn.Conv(n)d but uses `conv_layer` instead, for testing and as reference implementation.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            kernel_size (_type_): _description_
            stride (int, optional): _description_. Defaults to 1.
            padding (int, optional): _description_. Defaults to 0.
            dilation (int, optional): _description_. Defaults to 1.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool, optional): _description_. Defaults to True.
            padding_mode (str, optional): _description_. Defaults to 'zeros'.
            ndim (int, optional): _description_. Defaults to 2.

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
        """
        super().__init__()
        if groups != 1: raise NotImplementedError("groups != 1")
        if dilation != 1: raise NotImplementedError("dilation != 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # since padding is different in this implementation
        self.padding = padding * 2 if isinstance(padding, int) else [i*2 for i in padding]
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.ndim = ndim

        if isinstance(kernel_size, int): kernel_size = [kernel_size] * ndim

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size))
        torch.nn.init.kaiming_uniform_(self.weight, a = 5 ** 0.5)

        if bias: self.bias = torch.nn.Parameter(torch.randn(out_channels))
        else: self.bias = None

    def forward(self, x:torch.Tensor):
        return conv_layer(x, self.weight, self.bias, stride=self.stride, padding=self.padding, padding_mode=self.padding_mode)
