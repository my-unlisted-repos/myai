import typing
from collections import abc
import torch

def make_generic(m) -> None | abc.Callable[[torch.Tensor], torch.Tensor]:

    if m is None: return  None
    if isinstance(m, torch.nn.Module): return m
    if isinstance(m, type): return m()

    raise ValueError(m)