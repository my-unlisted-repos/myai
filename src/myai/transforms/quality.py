import random

import torch, numpy as np

from ._base import RandomTransform
from ..random import randmask

__all__ = [
    "add_gaussian_noise",
    "GaussianNoise",
    "add_gaussian_noise_triangular",
    "GaussianNoiseTriangular",
]
def add_gaussian_noise(x, alpha: float = 1):
    return x + torch.randn_like(x).mul_(alpha)

class GaussianNoise(RandomTransform):
    """GAUSSIAN NOISE WILL NOT BE AFFECTED BY SEED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
    def __init__(self, alpha:float = 1, p = 0.1, seed = None):
        super().__init__(seed)
        self.alpha = alpha
        self.p = p
    def forward(self, x): return add_gaussian_noise(x, self.alpha)

def add_gaussian_noise_triangular(x, low:float = 0, high: float = 1, mode: float | None = 0, ):
    return x + torch.randn_like(x) * random.triangular(low, high, mode)



class GaussianNoiseTriangular(RandomTransform):
    """GAUSSIAN NOISE WILL NOT BE AFFECTED BY SEED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
    def __init__(self, low:float = 0, high: float = 1, mode: float | None = 0, p = 0.1, seed = None):
        super().__init__(seed)
        self.p = p
        self.low=low;self.high=high;self.mode=mode

    def forward(self, x):
        return add_gaussian_noise_triangular(x, self.low, self.high, self.mode)
