from collections.abc import Callable
import typing
# pylint:disable=not-callable
import torch


class I1e(torch.nn.Module):
    """This worked really well in a single image autoencoding so I need to test this. It's also definitely quite slow."""
    def forward(self, x: torch.Tensor): return torch.special.i1e(x)

class Sine(torch.nn.Module):
    def __init__(self, mul = None):
        """computes `sin(x * mul)`"""
        super().__init__()
        self.mul = mul
    def forward(self, x: torch.Tensor):
        if self.mul is not None: x = x * self.mul
        return torch.sin(x)

class Modulus(torch.nn.Module):
    """computes `abs(x * mul)`"""
    def forward(self, x: torch.Tensor):
        return x.abs()

def make_activation(
    m,
    in_channels: int | None = None,
    ndim: int | None = None,
) -> None | Callable[[torch.Tensor], torch.Tensor]:
    if m is None: return None
    if isinstance(m, torch.nn.Module): return m
    elif isinstance(m, type): return m()

    if isinstance(m, str):
        m = m.strip().lower()
        if m == 'relu': return torch.nn.ReLU(True)
        if m == 'gelu': return torch.nn.GELU()
        if m == 'softmax': return torch.nn.Softmax(dim=1)
        if m == 'sigmoid': return torch.nn.Sigmoid()
        if m == 'identity': return torch.nn.Identity()
        if m == 'tanh': return torch.nn.Tanh()
        if m in ('sin', 'sine'): return Sine()
        if m == 'modulus': return Modulus()

    raise ValueError(m)

