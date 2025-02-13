"img oladers "

import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import skimage.io
import torch
import torchvision.io
import torchvision.transforms.v2
from imageio import v3

from ..transforms import intensity, force_hw3, tonumpy


def imread_skimage(path:str) -> np.ndarray:
    return skimage.io.imread(path)

def imread_plt(path:str) -> np.ndarray:
    return plt.imread(path)

def imread_cv2(path):
    return cv2.imread(path)[:, :, ::-1] # BRG -> RGB # pylint:disable=no-member

def imread_imageio(path):
    return v3.imread(path)

def imread_pil(path:str) -> np.ndarray:
    return np.array(PIL.Image.open(path))

def imreadtensor_torchvision(path:str, dtype=None, device=None) -> torch.Tensor:
    return torchvision.io.read_image(path).to(dtype=dtype, device=device, copy=False)

def imreadtensor_skimage(path:str) -> torch.Tensor:
    return torch.as_tensor(imread_skimage(path))

def imreadtensor_plt(path:str) -> torch.Tensor:
    return torch.as_tensor(imread_plt(path))

def imreadtensor_pil(path:str) -> torch.Tensor:
    return torch.as_tensor(imread_pil(path))

def imread(path, fns = (imread_pil, imread_plt, imread_cv2, imread_skimage, imread_imageio, )) -> np.ndarray:
    """returns channel last numpy array"""
    for i,f in enumerate(fns):
        try: return f(path)
        except Exception as e:
            if i == len(fns) - 1: raise e
    raise ValueError("Could not read image")

def imreadtensor(path:str, dtype=None, device=None):
    """returns channel first tensor"""
    if path.lower().endswith(('jpg', 'jpeg', 'png', 'gif')):
        try:
            return imreadtensor_torchvision(path, dtype=dtype, device=device)
        except Exception:
            return torch.as_tensor(imread(path), dtype=dtype, device=device).moveaxis(-1, 0)
    else:
        return torch.as_tensor(imread(path), dtype=dtype, device=device).moveaxis(-1, 0)

def imwrite(x:np.ndarray | torch.Tensor, outfile:str, mkdir=False, normalize=True, compression = 9, optimize=True):
    """if normalize is False, x must be a UINT8 IMAGE WITH VALUES FROM 0 TO 255"""
    x = tonumpy(force_hw3(x))
    if normalize: x = intensity.normalize(x, 0, 255).astype(np.uint8) # type:ignore
    if mkdir and not os.path.exists(os.path.dirname(outfile)): os.mkdir(os.path.dirname(outfile))
    PIL.Image.fromarray(x).save(outfile, optimize=optimize, compress_level=compression) # type:ignore
