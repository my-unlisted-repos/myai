import typing as T
from typing import Any

import numpy as np
import torch

from ._base import Transform


@T.overload
def force_hwc(x: torch.Tensor) -> torch.Tensor: ...
@T.overload
def force_hwc(x: np.ndarray) -> np.ndarray: ...
def force_hwc(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Forces input tensor to be (H, W, C) format.
    The smallest out of first, last, third last dimensions is assumed to be the channel dimension!
    So that would work on (B, C, H, W); (C, D, H, W), and (D, H, W, C).
    If tensor has 2 channels, the third will be filled with zeroes.
    Takes the central slice of tensors with more than 3 dimensions."""
    x = x.squeeze()

    while x.ndim > 3:

        # first dimension is channel dimension, move it last
        if x.shape[0] < x.shape[-1] and x.shape[0] < x.shape[-3]:
            if isinstance(x, torch.Tensor): x = x.moveaxis(0, -1)
            else: x = np.moveaxis(x, 0, -1)

        # first dimension is not channel dimension, take central slice
        else:
            x = x[x.shape[0] // 2]

    # create channel dimension if it doesn't exist
    if x.ndim == 2: x = x[:,:,None]

    # channel first to channel last
    if x.shape[0] < x.shape[-1]:
        if isinstance(x, torch.Tensor): x = x.moveaxis(0, -1)
        else: x = np.moveaxis(x, 0, -1)

    return x

@T.overload
def force_hw3(x: torch.Tensor, allow_4_channels: bool = False) -> torch.Tensor: ...
@T.overload
def force_hw3(x: np.ndarray, allow_4_channels: bool = False) -> np.ndarray: ...
def force_hw3(x: torch.Tensor | np.ndarray, allow_4_channels = False) -> torch.Tensor | np.ndarray:
    """Forces input tensor to be (H, W, 3) format.
    If tensor has 2 channels, the third will be filled with zeroes.
    Takes the central slice of tensors with more than 3 dimensions."""

    x = force_hwc(x)
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
        x = x[:,:,:maxv]
    return x

@T.overload
def force_hw(x: torch.Tensor) -> torch.Tensor: ...
@T.overload
def force_hw(x: np.ndarray) -> np.ndarray: ...
def force_hw(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Forces input tensor to be (H, W) format.
    Averages over the channel dimension.
    Takes the central slice of tensors with more than 3 dimensions."""
    return force_hwc(x).mean(-1)


def ensure_tensor(x:Any, device = None, dtype = None) -> torch.Tensor:
    """If x is not a tensor, uses torch.as_tensor."""
    if isinstance(x, torch.Tensor): return x.to(device=device, dtype=dtype, copy=False)
    else: return torch.as_tensor(x, device=device, dtype=dtype)

class EnsureTensor(Transform):
    def __init__(self, device = None, dtype = None):
        self.device = device
        self.dtype = dtype
    def forward(self, x:Any) -> torch.Tensor:
        return ensure_tensor(x, device=self.device, dtype=self.dtype)

def ensure_dtype(x:torch.Tensor, dtype:torch.dtype) -> torch.Tensor:
    return x.to(dtype)

class EnsureDtype(Transform):
    def __init__(self, dtype:torch.dtype):
        self.dtype = dtype
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return ensure_dtype(x, dtype=self.dtype)

def ensure_device(x:torch.Tensor, device:torch.device) -> torch.Tensor:
    return x.to(device)

class EnsureDevice(Transform):
    def __init__(self, device:torch.device):
        self.device = device
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return ensure_device(x, device=self.device)

def ensure_channel_first(x:torch.Tensor) -> torch.Tensor:
    """x must be `(C, *)` or `(*, C)`. This ensures `(C, *)`"""
    if x.shape[0] > x.shape[-1]: return x.movedim(-1, 0)
    return x

def ensure_channel_last(x:torch.Tensor) -> torch.Tensor:
    """x must be `(C, *)` or `(*, C)`. This ensures `(*, C)`"""
    if x.shape[0] < x.shape[-1]: return x.movedim(0, -1)
    return x
