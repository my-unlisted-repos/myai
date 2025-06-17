"img oladers "
import importlib.util
import logging
import os
from typing import Any, cast

import numpy as np
import PIL.Image
import torch
import torchvision.io

if importlib.util.find_spec('cv2') is not None: import cv2 # pylint: disable=import-error # type:ignore
else: cv2 = cast(Any, None)

if importlib.util.find_spec('skimage') is not None: import skimage.io as skimage_io # pylint: disable=import-error # type:ignore
else: skimage_io = cast(Any, None)

if importlib.util.find_spec('matplotlib') is not None: import matplotlib.pyplot as plt
else: plt = cast(Any, None)

if importlib.util.find_spec('imageio') is not None: from imageio import v3 # pylint: disable=import-error # type:ignore
else: v3 = cast(Any, None)



def _imread_skimage(path:str) -> np.ndarray:
    return skimage_io.imread(path)

def _imread_plt(path:str) -> np.ndarray:
    return plt.imread(path)

def _imread_cv2(path):
    return cv2.imread(path)[:, :, ::-1] # BRG -> RGB # pylint:disable=no-member

def _imread_imageio(path):
    return v3.imread(path)

def _imread_pil(path:str) -> np.ndarray:
    return np.array(PIL.Image.open(path))

def _imread_torchvision(path:str, dtype=None, device=None) -> torch.Tensor:
    return torchvision.io.read_image(path).to(dtype=dtype, device=device, copy=False)

def imread(path, fns = (_imread_pil, _imread_plt, _imread_cv2, _imread_skimage, _imread_imageio, )) -> np.ndarray:
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
            return _imread_torchvision(path, dtype=dtype, device=device)
        except Exception:
            return torch.as_tensor(imread(path), dtype=dtype, device=device).moveaxis(-1, 0)
    else:
        return torch.as_tensor(imread(path), dtype=dtype, device=device).moveaxis(-1, 0)

def imwrite(x:np.ndarray | torch.Tensor, outfile:str, mkdir=False, normalize=True, compression = 9, optimize=True):
    """if normalize is False, x must be a UINT8 IMAGE WITH VALUES FROM 0 TO 255"""
    from ..transforms import to_HW3, intensity, tonumpy
    x = tonumpy(to_HW3(x))
    if normalize: x = intensity.normalize(x, 0, 255).astype(np.uint8) # type:ignore
    if mkdir and not os.path.exists(os.path.dirname(outfile)): os.mkdir(os.path.dirname(outfile))
    PIL.Image.fromarray(x).save(outfile, optimize=optimize, compress_level=compression) # type:ignore
