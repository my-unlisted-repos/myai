import random
from collections.abc import Sequence

import torch
from torchvision.transforms import v2

from ._base import RandomTransform

__all__ = [
    "randflip",
    "randflipt",
    "RandFlip",
    "RandFlipt",
    "randrot90",
    "randrot90t",
    "RandRot90",
    "RandRot90t",
    "fast_slice_reduce_size",
    "FastSliceReduceSize",
    "resize_to_fit",
    "resize_to_contain",
]
def randflip(x:torch.Tensor):
    flip_dims = random.sample(population = range(1, x.ndim), k = random.randint(1, x.ndim-1))
    return x.flip(flip_dims)

def randflipt(x:Sequence[torch.Tensor]):
    flip_dims = random.sample(population = range(1, x[0].ndim), k = random.randint(1, x[0].ndim-1))
    return [i.flip(flip_dims) for i in x]

class RandFlip(RandomTransform):
    def __init__(self, p:float=0.5, seed = None):
        super().__init__(seed)
        self.p = p
    def forward(self, x:torch.Tensor): return randflip(x)

class RandFlipt(RandomTransform):
    def __init__(self, p:float=0.5, seed = None):
        super().__init__(seed)
        self.p = p
    def forward(self, x:Sequence[torch.Tensor]): return randflipt(x)

def randrot90(x:torch.Tensor):
    flip_dims = random.sample(range(1, x.ndim), k=2)
    k = random.randint(-3, 3)
    return x.rot90(k = k, dims = flip_dims)

def randrot90t(x:Sequence[torch.Tensor]):
    flip_dims = random.sample(range(1, x[0].ndim), k=2)
    k = random.randint(-3, 3)
    return [i.rot90(k = k, dims = flip_dims) for i in x]

class RandRot90(RandomTransform):
    def __init__(self, p:float=0.5, seed = None):
        super().__init__(seed)
        self.p = p
    def forward(self, x:torch.Tensor): return randrot90(x)

class RandRot90t(RandomTransform):
    def __init__(self, p:float=0.5, seed = None):
        super().__init__(seed)
        self.p = p
    def forward(self, x:Sequence[torch.Tensor]): return randrot90t(x)


def fast_slice_reduce_size(x:torch.Tensor, min_shape: Sequence[int]):
    times = [i/j for i,j in zip(x.shape[1:], min_shape)]
    min_times = int(min(times))
    if min_times <= 2:
        return x
    reduction = random.randrange(2, min_times)
    ndim = x.ndim
    if ndim == 2: return x[:, ::reduction]
    if ndim == 3: return x[:, ::reduction, ::reduction]
    if ndim == 4: return x[:, ::reduction, ::reduction, ::reduction]
    raise ValueError(f'{x.shape = }')

def fast_slice_reduce_sizet(seq:Sequence[torch.Tensor], min_shape: Sequence[int]):
    x = seq[0]
    times = [i/j for i,j in zip(x.shape[1:], min_shape)]
    min_times = int(min(times))
    if min_times <= 2:
        return seq
    reduction = random.randrange(2, min_times)
    ndim = x.ndim
    if ndim == 2: return [i[:, ::reduction] for i in seq]
    if ndim == 3: return [i[:, ::reduction, ::reduction] for i in seq]
    if ndim == 4: return [i[:, ::reduction, ::reduction, ::reduction] for i in seq]
    raise ValueError(f'{x.shape = }')

class FastSliceReduceSize(RandomTransform):
    def __init__(self, min_shape: Sequence[int], p:float=0.5, seed = None):
        super().__init__(seed)
        self.min_shape = min_shape
        self.p = p
    def forward(self, x:torch.Tensor): return fast_slice_reduce_size(x, self.min_shape)

class FastSliceReduceSizet(RandomTransform):
    def __init__(self, min_shape: Sequence[int], p:float=0.5, seed = None):
        super().__init__(seed)
        self.min_shape = min_shape
        self.p = p
    def forward(self, x:Sequence[torch.Tensor]): return fast_slice_reduce_sizet(x, self.min_shape)


def resize_to_fit(
    image: torch.Tensor,
    size: int,
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
    antialias=True,
):
    """Resize to fit into a (size x size) square. Image must be (C, H, W). Resizes so that larger side is `size`."""
    largest = max(image.shape[1:])
    factor = size / largest
    return v2.functional.resize(
        image,
        size = [min(int(image.shape[1]*factor), size), min(int(image.shape[2]*factor), size)],
        interpolation=interpolation,
        antialias=antialias,
    )

def resize_to_contain(
    image: torch.Tensor,
    size: int,
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
    antialias=True,
):
    """Resize to contain a (size x size) square. Image must be (C, H, W). Resizes so that smaller side is `size`."""
    smallest = min(image.shape[1:])
    factor = size / smallest

    if image.shape[1] >= image.shape[2]:
        imsize =  [int(image.shape[1]*factor), size]
    else:
        imsize = [size, int(image.shape[2]*factor)]
    return v2.functional.resize(
        image,
        size = imsize,
        interpolation=interpolation,
        antialias=antialias,
    )
