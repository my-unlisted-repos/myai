# pylint:disable=not-callable
import collections.abc as A
import typing as T
import torch
from .conv import convnd, ConvLike
from .convtranspose import ConvTransposeLike, convtransposend

class RFFT(torch.nn.Module):
    """Computes the N-dimensional discrete Fourier transform of real input"""
    def __init__(self, dim = (-2, -1), norm: T.Literal['forward', 'backward', 'ortho'] = 'backward'):
        super().__init__()
        self.dim = dim
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfftn(x, dim = self.dim, s = [x.shape[i] for i in self.dim], norm = self.norm)

class FFT(torch.nn.Module):
    """Computes the N-dimensional discrete Fourier transform of real input"""
    def __init__(self, dim = (-2, -1), norm: T.Literal['forward', 'backward', 'ortho'] = 'backward'):
        super().__init__()
        self.dim = dim
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftn(x, dim = self.dim, s = [x.shape[i] for i in self.dim], norm = self.norm)

class IRFFT(torch.nn.Module):
    """Computes the inverse of RFFT."""
    def __init__(self, dim = (-2, -1), norm: T.Literal['forward', 'backward', 'ortho'] = 'backward'):
        super().__init__()
        self.dim = dim
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfftn(x, dim = self.dim, norm = self.norm)
    
class IFFT(torch.nn.Module):
    """Computes the inverse of RFFT."""
    def __init__(self, dim = (-2, -1), norm: T.Literal['forward', 'backward', 'ortho'] = 'backward'):
        super().__init__()
        self.dim = dim
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfftn(x, dim = self.dim, norm = self.norm)


class RFFTSwap(torch.nn.Module):
    """Swap real and imaginary parts..."""
    def __init__(self, dim = (-2, -1), norm: T.Literal['forward', 'backward', 'ortho'] = 'backward'):
        super().__init__()
        self.dim = dim
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.rfftn(x, dim = self.dim, s = [x.shape[i] for i in self.dim], norm = self.norm)
        return torch.fft.irfftn(torch.complex(x.imag, x.real), dim = self.dim, )

class RFFTBlock(torch.nn.Module):
    def __init__(
        self,
        module: A.Callable[[torch.Tensor], torch.Tensor],
        dim=(-2, -1),
        norm: T.Literal["forward", "backward", "ortho"] = "backward",
    ):
        super().__init__()
        self.module = module
        self.dim = dim
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.rfftn(x, dim = self.dim, s = [x.shape[i] for i in self.dim], norm = self.norm)
        x = self.module(x)
        return torch.fft.irfftn(x, dim = self.dim, norm = self.norm)

class SplitRFFTBlock(torch.nn.Module):
    def __init__(
        self,
        module_real: A.Callable[[torch.Tensor], torch.Tensor],
        module_imag: A.Callable[[torch.Tensor], torch.Tensor],
        dim=(-2, -1),
        norm: T.Literal["forward", "backward", "ortho"] = "backward",
    ):
        super().__init__()
        self.module_real = module_real
        self.module_imag = module_imag
        self.dim = dim
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.rfftn(x, dim = self.dim, s = [x.shape[i] for i in self.dim], norm = self.norm)
        x = torch.complex(self.module_real(x.real), self.module_imag(x.imag))
        return torch.fft.irfftn(x, dim = self.dim, norm = self.norm)

class InverseRFFTBlock(torch.nn.Module):
    def __init__(
        self,
        module: A.Callable[[torch.Tensor], torch.Tensor],
        dim=(-2, -1),
        norm: T.Literal["forward", "backward", "ortho"] = "backward",
        ret = 'imag',
    ):
        super().__init__()
        self.module = module
        self.dim = dim
        self.norm = norm
        self.ret = ret

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.irfftn(x, dim = self.dim, norm = self.norm)
        x = self.module(x)
        x = torch.fft.rfftn(x, dim = self.dim, s = [x.shape[i] for i in self.dim], norm = self.norm)
        if self.ret == 'real': return x.real
        return x.imag


class RFFTConv(RFFTBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        dtype = None, # not used but needed for protocol
        ndim = 2,
        conv_module: ConvLike = convnd,
        fft_dim=(-2, -1),
        norm: T.Literal["forward", "backward", "ortho"] = "backward",
    ):
        super().__init__(
            conv_module(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
                ndim=ndim,
                dtype=torch.complex64,
            ),
            dim = fft_dim,
            norm = norm,
        )

__test_rfftconv: ConvLike = RFFTConv

class SplitRFFTConv(SplitRFFTBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        dtype = None,
        ndim = 2,
        conv_module: ConvLike = convnd,
        fft_dim=(-2, -1),
        norm: T.Literal["forward", "backward", "ortho"] = "backward",
    ):
        convs = [
            conv_module(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
                ndim=ndim,
                dtype=dtype,
            )
            for _ in range(2)
        ]
        super().__init__(*convs, dim = fft_dim, norm = norm)

__test_split_rfftconv: ConvLike = SplitRFFTConv


class RFFTConvTranspose(RFFTBlock):
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
        dtype = None, # not used
        ndim: int = 2,
        conv_module: ConvTransposeLike = convtransposend,
        fft_dim=(-2, -1),
        norm: T.Literal["forward", "backward", "ortho"] = "backward",
    ):
        super().__init__(
            conv_module(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
                ndim=ndim,
                dtype=torch.complex64,
            ),
            dim = fft_dim,
            norm = norm,
        )

__test_rfftconvtranspose: ConvTransposeLike = RFFTConvTranspose

class SplitRFFTConvTranspose(SplitRFFTBlock):
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
        conv_module: ConvTransposeLike = convtransposend,
        fft_dim=(-2, -1),
        norm: T.Literal["forward", "backward", "ortho"] = "backward",
    ):
        convs = [
            conv_module(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding = output_padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
                ndim=ndim,
                dtype=dtype,
            )
            for _ in range(2)
        ]
        super().__init__(*convs, dim = fft_dim, norm = norm)

__test_split_rfftconvtranspose: ConvTransposeLike = SplitRFFTConvTranspose
