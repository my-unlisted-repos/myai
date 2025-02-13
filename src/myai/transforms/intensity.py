import random
from typing import overload

import numpy as np
import torch
from torchzero.random import uniform
from ._base import RandomTransform, Transform


__all__ = [
    "znormalize",
    'ZNormalize',
    "rand_znormalize",
    'RandZNormalize',
    "znormalize_ch",
    "unznormalize_ch",
    'ZNormalizeCh',
    "rand_znormalize_ch",
    'RandZNormalizeCh',
    "meanstd_normalize_ch",
    "znormalize_batch",
    "ZNormalizeBatch",
    "rand_znormalize_batch",
    "RandZNormalizeBatch",
    "normalize",
    "Normalize",
    "rand_shift",
    "RandShift",
    "rand_scale",
    "RandScale",
    "normalize_ch",
    "NormalizeCh",
    "normalize_batch",
    "NormalizeBatch",
    "shrink",
    "Shrink",
    "rand_shrink",
    "RandShrink",
    "contrast",
    "Contrast",
    "rand_contrast",
    "RandContrast",
]

@overload
def znormalize(x:torch.Tensor, mean=0., std=1.) -> torch.Tensor: ...
@overload
def znormalize(x:np.ndarray, mean=0., std=1.) -> np.ndarray: ...
def znormalize(x:torch.Tensor | np.ndarray, mean=0., std=1.) -> torch.Tensor | np.ndarray:
    """Global z-normalization"""
    xstd = x.std()
    if xstd != 0: return ((x - x.mean()) / (xstd / std)) + mean
    return x - x.mean()

class ZNormalize(Transform):
    def __init__(self, mean=0., std=1.):
        """Global z-normalization"""
        self.mean = mean
        self.std = std
    def forward(self, x): return znormalize(x, self.mean, self.std)

def rand_znormalize(x, mean = (-1., 1.), std = (0.5, 2)):
    meanv = random.uniform(*mean)
    stdv = random.uniform(*std)
    return znormalize(x, meanv, stdv)

class RandZNormalize(RandomTransform):
    def __init__(self, mean = (-1., 1.), std = (0.5, 2), p=0.1, seed = None):
        super().__init__(seed)
        self.mean = mean
        self.std = std
        self.p = p
    def forward(self, x): return rand_znormalize(x, self.mean, self.std)


def znormalize_ch(x:torch.Tensor, mean=0., std=1.):
    """channel-wise Z-normalization"""
    std = x.std(list(range(1, x.ndim)), keepdim = True) / std
    std[std==0] = 1
    return ((x - x.mean(list(range(1, x.ndim)), keepdim=True)) / std) + mean

def meanstd_normalize_ch(x:torch.Tensor, mean, std):
    """Normalize to mean 0 std 1 using given mean and std values"""
    return ((x - x.mean(list(range(1, x.ndim)), keepdim=True)) / std) + mean

def unznormalize_ch(x, mean, std):
    """Undoes channel-wise z-norm"""
    inverse_mean = [-mean[i]/std[i] for i in range(len(mean))]
    inverse_std = [1/std[i] for i in range(len(std))]
    return meanstd_normalize_ch(x, inverse_mean, inverse_std)


class ZNormalizeCh(Transform):
    def __init__(self, mean=0., std=1.):
        """channel-wise Z-normalization"""
        self.mean = mean
        self.std = std
    def forward(self, x): return znormalize_ch(x, self.mean, self.std)

def rand_znormalize_ch(x, mean = (-1., 1.), std = (0.5, 2)):
    meanv = random.uniform(*mean)
    stdv = random.uniform(*std)
    return znormalize_ch(x, meanv, stdv)

class RandZNormalizeCh(RandomTransform):
    def __init__(self, mean = (-1., 1.), std = (0.5, 2), p=0.1, seed = None):
        super().__init__(seed)
        self.mean = mean
        self.std = std
        self.p = p
    def forward(self, x): return rand_znormalize_ch(x, self.mean, self.std)


def znormalize_batch(x:torch.Tensor, mean=0., std=1.):
    """z-normalize a batch channel-wise"""
    std = x.std(list(range(2, x.ndim)), keepdim = True) / std
    std[std==0] = 1
    return ((x - x.mean(list(range(2, x.ndim)), keepdim=True)) / std) + mean

class ZNormalizeBatch(Transform):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    def forward(self, x): return znormalize_batch(x, self.mean, self.std)

def rand_znormalize_batch(x, mean = (-1., 1.), std = (0.5, 2)):
    meanv = random.uniform(*mean)
    stdv = random.uniform(*std)
    return znormalize_batch(x, meanv, stdv)

class RandZNormalizeBatch(RandomTransform):
    def __init__(self, mean = (-1., 1.), std = (0.5, 2), p=0.1, seed = None):
        super().__init__(seed)
        self.mean = mean
        self.std = std
        self.p = p
    def forward(self, x): return rand_znormalize_batch(x, self.mean, self.std)

@overload
def normalize(x:torch.Tensor, min:float|torch.Tensor=0., max:float|torch.Tensor=1.) -> torch.Tensor: ...
@overload
def normalize(x:np.ndarray, min:float|np.ndarray=0., max:float|np.ndarray=1.) -> np.ndarray: ...
def normalize(x:torch.Tensor | np.ndarray, min:float|np.ndarray|torch.Tensor=0., max:float|np.ndarray|torch.Tensor=1.) -> torch.Tensor | np.ndarray:
    """Normalize to `[min, max]`"""
    x = x - x.min()
    xmax = x.max()
    if xmax != 0: x = x / xmax
    else: return x
    return x * (max - min) + min

class Normalize(Transform):
    def __init__(self, min=0, max=1):
        """Normalize to `[min, max]`"""
        self.min = min
        self.max = max
    def forward(self, x): return normalize(x, self.min, self.max)

def rand_shift(x:torch.Tensor | np.ndarray, val = (-1., 1.)):
    """Shift the input by a random amount. """
    return x + random.uniform(*val)

class RandShift(RandomTransform):
    def __init__(self, val = (-1., 1.), p=0.1, seed = None):
        super().__init__(seed)
        self.val = val
        self.p = p
    def forward(self, x): return rand_shift(x, self.val)

@overload
def rand_scale(x:torch.Tensor, val = (0.5, 2)) -> torch.Tensor: ...
@overload
def rand_scale(x:np.ndarray, val = (0.5, 2)) -> np.ndarray: ...
def rand_scale(x:torch.Tensor | np.ndarray, val = (0.5, 2)) -> torch.Tensor | np.ndarray:
    """Scale the input by a random amount. """
    return x * random.uniform(*val)

class RandScale(RandomTransform):
    def __init__(self, val = (0.5, 2), p=0.1, seed = None):
        super().__init__(seed)
        self.val = val
        self.p = p
    def forward(self, x): return rand_scale(x, self.val)

def normalize_ch(x:torch.Tensor, min=0, max=1): #pylint:disable=W0622
    """Normalize to `[min, max]` channel-wise"""
    x = x - x.amin(list(range(1, x.ndim)), keepdim = True)
    xmax = x.amax(list(range(1, x.ndim)), keepdim = True)
    xmax[xmax==0] = 1
    return (x / xmax) * (max - min) + min

class NormalizeCh(Transform):
    def __init__(self, min=0, max=1):
        """Normalize to `[min, max]` channel-wise"""
        self.min = min
        self.max = max
    def forward(self, x): return normalize_ch(x, self.min, self.max)

def normalize_batch(x:torch.Tensor, min=0, max=1): #pylint:disable=W0622
    """Normalize to `[min, max]` channel-wise"""
    x = x - x.amin(list(range(2, x.ndim)), keepdim = True)
    xmax = x.amax(list(range(2, x.ndim)), keepdim = True)
    xmax[xmax==0] = 1
    return (x / xmax) * (max - min) + min


class NormalizeBatch(Transform):
    def __init__(self, min=0, max=1):
        """Normalize to `[min, max]` channel-wise"""
        self.min = min
        self.max = max
    def forward(self, x): return normalize_batch(x, self.min, self.max)

def shrink(x:np.ndarray | torch.Tensor, min=0.2, max=0.8):
    """Shrink the range of the input"""
    xmin = x.min()
    xmax = x.max()
    r = xmax - xmin
    return x.clip(xmin + r * min, xmax - r * (1-max))

class Shrink(Transform):
    def __init__(self, min=0.2, max=0.8):
        """Shrink the range of the input"""
        self.min = min
        self.max = max
    def forward(self, x): return shrink(x, self.min, self.max)

def rand_shrink(x:np.ndarray | torch.Tensor, min=(0., 0.45), max=(0.55, 1.)):
    """Shrink the range of the input"""
    minv = random.uniform(*min)
    maxv = random.uniform(*max)
    return shrink(x, minv, maxv)

class RandShrink(RandomTransform):
    def __init__(self, min=(0., 0.45), max=(0.55, 1.), p=0.1, seed = None):
        """Shrink the range of the input"""
        super().__init__(seed)
        self.min = min
        self.max = max
        self.p = p
    def forward(self, x): return rand_shrink(x, self.min, self.max)

def contrast(x: torch.Tensor, min=0.2, max=0.8):
    """Shrink the range of the input and expand back to original range"""
    xmin = x.min()
    xmax = x.max()
    r = xmax - xmin
    return normalize(x.clamp(xmin + r * min, xmax - r * (1-max)), xmin, xmax)

class Contrast(Transform):
    def __init__(self, min=0.2, max=0.8):
        """Shrink the range of the input and expand back to original range"""
        self.min = min
        self.max = max
    def forward(self, x): return contrast(x, self.min, self.max)

def rand_contrast(x, min=(0., 0.45), max=(0.55, 1.)):
    """Shrink the range of the input and expand back to original range"""
    minv = random.uniform(*min)
    maxv = random.uniform(*max)
    return contrast(x, minv, maxv)

class RandContrast(RandomTransform):
    def __init__(self, min=(0., 0.45), max=(0.55, 1.), p=0.1, seed = None):
        """Shrink the range of the input and expand back to original range"""
        super().__init__(seed)
        self.min = min
        self.max = max
        self.p = p
    def forward(self, x): return rand_contrast(x, self.min, self.max)
