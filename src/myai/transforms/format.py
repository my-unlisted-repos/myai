from collections import abc
from typing import Any, overload

import numpy as np
import torch

from ._base import Transform
from ..loaders import generic

tositk = generic.tositk

def totensor(x:Any, device = None, dtype = None) -> torch.Tensor:
    """If x is not a tensor, uses torch.as_tensor(np.array(x))."""
    if isinstance(x, str): x = generic.read(x)
    if generic.is_sitk(x): x = generic.sitk_to_numpy(x)

    if isinstance(x, torch.Tensor): return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)

def tonumpy(x) -> np.ndarray:
    """Converts x to numpy array if it is not already one. If tensor, ensures it is detached and on CPU."""
    if isinstance(x, str): x = generic.read(x)
    if generic.is_sitk(x): return generic.sitk_to_numpy(x)

    if isinstance(x, np.ndarray): return x
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)

def tofloat(x) -> float:
    if isinstance(x, torch.Tensor): return float(x.detach().cpu().item())
    if isinstance(x, np.ndarray): return float(x.item())
    return float(x)


@overload
def to_HWC(x: torch.Tensor, strict: bool=False) -> torch.Tensor: ...
@overload
def to_HWC(x: np.ndarray, strict: bool=False) -> np.ndarray: ...
def to_HWC(x: torch.Tensor | np.ndarray, strict: bool=False) -> torch.Tensor | np.ndarray:
    """Forces input tensor to be (H, W, C) format.

    If tensor has 2 channels, the third will be filled with zeroes.
    Takes the central slice of tensors with more than 3 dimensions."""
    if x.ndim < 2: raise ValueError(f'to_HWC got a 1D or 0D tensor of shape {x.shape}')

    while x.ndim > 3:
        if strict:
            raise ValueError(f"to_HWC got a tensor of invalid shape {x.shape}")

        # first dimension is not channel dimension, take central slice
        if x.shape[0] == 1: x = x[0]
        else: x = x[x.shape[0] // 2]

    # create channel dimension if it doesn't exist
    if x.ndim == 2: x = x[:,:,None]

    # channel first to channel last
    if x.shape[0] < x.shape[-1]:
        if isinstance(x, torch.Tensor): x = x.moveaxis(0, -1)
        else: x = np.moveaxis(x, 0, -1)

    return x

@overload
def to_HW3(x: torch.Tensor, allow_4_channels: bool = False, strict: bool=False) -> torch.Tensor: ...
@overload
def to_HW3(x: np.ndarray, allow_4_channels: bool = False, strict: bool=False) -> np.ndarray: ...
def to_HW3(x: torch.Tensor | np.ndarray, allow_4_channels = False, strict: bool=False) -> torch.Tensor | np.ndarray:
    """Forces input tensor to be (H, W, 3) format.
    If tensor has 2 channels, the third will be filled with zeroes.
    Takes the central slice of tensors with more than 3 dimensions."""

    x = to_HWC(x, strict=strict)
    maxv = 4 if allow_4_channels else 3

    # (H, W, 1), repeat a single channel 3 times
    if x.shape[-1] == 1:
        if isinstance(x, torch.Tensor): x = torch.cat([x,x,x], dim=-1)
        else: x = np.concatenate([x,x,x], axis=-1)
    # (H, W, 2), add the third channel with zeroes
    elif x.shape[-1] == 2:
        if isinstance(x, torch.Tensor): x = torch.cat([x, torch.zeros_like(x[:,:,0,None])], dim=-1)
        else: x = np.concatenate([x, np.zeros_like(x[:,:,0,None])], axis=-1)
    # (H, W, 4+), remove extra channels
    elif x.shape[-1] > maxv:
        if strict: raise ValueError(f"to_HW3 got a tensor with more than {maxv} channels: {x.shape = }")
        x = x[:,:,:maxv]
    return x

@overload
def to_HW(x: torch.Tensor, strict: bool=False) -> torch.Tensor: ...
@overload
def to_HW(x: np.ndarray, strict: bool=False) -> np.ndarray: ...
def to_HW(x: torch.Tensor | np.ndarray, strict: bool=False) -> torch.Tensor | np.ndarray:
    """Forces input tensor to be (H, W) format.
    Averages over the channel dimension.
    Takes the central slice of tensors with more than 3 dimensions."""
    if x.ndim == 2: return x
    return to_HWC(x, strict=strict).mean(-1)


@overload
def to_CHW(x: torch.Tensor, strict: bool=False) -> torch.Tensor: ...
@overload
def to_CHW(x: np.ndarray, strict: bool=False) -> np.ndarray: ...
def to_CHW(x: torch.Tensor | np.ndarray, strict: bool=False) -> torch.Tensor | np.ndarray:
    x = to_HWC(x, strict=strict)
    if isinstance(x, torch.Tensor): return x.moveaxis(-1, 0)
    return np.moveaxis(x, -1, 0)

@overload
def to_3HW(x: torch.Tensor, allow_4_channels: bool = False, strict: bool=False) -> torch.Tensor: ...
@overload
def to_3HW(x: np.ndarray, allow_4_channels: bool = False, strict: bool=False) -> np.ndarray: ...
def to_3HW(x: torch.Tensor | np.ndarray, allow_4_channels: bool = False, strict: bool=False) -> torch.Tensor | np.ndarray:
    x = to_HW3(x, allow_4_channels=allow_4_channels, strict=strict)
    if isinstance(x, torch.Tensor): return x.moveaxis(-1, 0)
    return np.moveaxis(x, -1, 0)


def to_channel_first(x:torch.Tensor) -> torch.Tensor:
    """x must be `(C, *)` or `(*, C)`. This ensures `(C, *)`"""
    if x.shape[0] > x.shape[-1]: return x.movedim(-1, 0)
    return x

def to_channel_last(x:torch.Tensor) -> torch.Tensor:
    """x must be `(C, *)` or `(*, C)`. This ensures `(*, C)`"""
    if x.shape[0] < x.shape[-1]: return x.movedim(0, -1)
    return x


class ToTensor(Transform):
    def __init__(self, device = None, dtype = None):
        self.device = device
        self.dtype = dtype
    def forward(self, x:Any) -> torch.Tensor:
        return totensor(x, device=self.device, dtype=self.dtype)

class ToDtype(Transform):
    def __init__(self, dtype:torch.dtype): self.dtype = dtype
    def forward(self, x:torch.Tensor) -> torch.Tensor: return x.to(dtype=self.dtype)

class ToDevice(Transform):
    def __init__(self, device:torch.device): self.device = device
    def forward(self, x:torch.Tensor) -> torch.Tensor: return x.to(device=self.device)


@overload
def maybe_detach_cpu(x: torch.Tensor) -> torch.Tensor: ...
@overload
def maybe_detach_cpu[X](x: X) -> X: ...
def maybe_detach_cpu[X](x: X) -> torch.Tensor | X:
    """If x is torch.tensor, ensures it is on CPU and detached. Otherwise returns x as is."""
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    return x

@overload
def to_numpy_or_none(x: torch.Tensor) -> np.ndarray: ...
@overload
def to_numpy_or_none(x: None) -> None: ...
@overload
def to_numpy_or_none(x: np.ndarray) -> np.ndarray: ...
def to_numpy_or_none(x: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    """Converts x to numpy array if it is not already one. If tensor, ensures it is detached and on CPU.
    If None, returns None."""
    if x is None: return None
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    return np.asarray(x)


@overload
def maybe_detach_cpu_recursive(x: torch.Tensor) -> torch.Tensor: ...
@overload
def maybe_detach_cpu_recursive[K](x: abc.Mapping[K, torch.Tensor | Any]) -> dict[K, torch.Tensor | Any]: ...
@overload
def maybe_detach_cpu_recursive(x: abc.MutableSequence[torch.Tensor | Any]) -> list[torch.Tensor | Any]: ...
@overload
def maybe_detach_cpu_recursive(x: tuple[torch.Tensor | Any]) -> tuple[torch.Tensor | Any]: ...
@overload
def maybe_detach_cpu_recursive[V](x: V) -> V: ...
def maybe_detach_cpu_recursive(x):
    """Recursively detaches and moves x or all elements in x to CPU. Can be slow!"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, abc.Mapping):
        return {k: maybe_detach_cpu_recursive(v) for k, v in x.items()}
    if isinstance(x, abc.MutableSequence):
        return [maybe_detach_cpu_recursive(v) for v in x]
    if isinstance(x, abc.Sequence):
        return tuple(maybe_detach_cpu_recursive(v) for v in x)
    return x

def to_numpy_recursive(x) -> np.ndarray:
    """Converts x to numpy array if it is not already one. If sequence of tensors, ensures it is detached and on CPU, RECURSIVELY. Can be slow!"""
    if isinstance(x, str): raise RuntimeError(f'to_numpy_recursive got string "{x}"')
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, abc.Sequence):
        return np.asarray([to_numpy_recursive(i) for i in x])
    return np.asarray(x)

def to_numpy_or_none_recursive(x) -> np.ndarray | None:
    """Converts x to numpy array if it is not already one. If tensor, ensures it is detached and on CPU, RECURSIVELY. Can be slow! If None, returns None."""
    if x is None: return None
    return to_numpy_recursive(x)
