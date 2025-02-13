# pylint:disable=not-callable
import collections.abc as A
import typing as T
import torch

class Pseudoinverse(torch.nn.Module):
    """Computes the pseudoinverse (Moore-Penrose inverse) of a matrix."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.pinv(x)

class PseudoinverseBlock(torch.nn.Module):
    def __init__(
        self,
        module: A.Callable[[torch.Tensor], torch.Tensor],
    ):
        
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.linalg.pinv(x)
        x = self.module(x)
        return torch.linalg.pinv(x)
