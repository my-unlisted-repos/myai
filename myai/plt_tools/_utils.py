import warnings

import numpy as np


def _prepare_image_for_plotting(x: np.ndarray, allow_4_channels: bool = False, warn = True) -> np.ndarray:
    # plt requirement: arrays must be of dtype byte, short, float32 or float64
    if x.dtype not in (np.float32, np.float64):
        if x.dtype == np.uint8: x = x.astype(np.float32) / 255
        else: x = x.astype(np.float32)

    x = x.squeeze()
    if x.ndim == 2: return x

    while x.ndim > 3:
        # color_dim = np.argsort(x.shape)[0]
        # if color_dim != x.ndim - 1: x = np.moveaxis(x, color_dim, -1)
        if x.shape[0] > 1:
            if warn: warnings.warn(f"Image has more than 3 dimensions, picking the central slice. Shape is {x.shape}")
        x = x[x.shape[0] // 2]

    if x.shape[0] < x.shape[-1]: x = np.moveaxis(x, 0, -1)

    maxv = 4 if allow_4_channels else 3
    if x.shape[-1] == 2: x = np.concatenate([x, np.zeros_like(x[:,:,0,None])], axis=-1)
    elif x.shape[-1] > maxv:
        if warn: warnings.warn(f'Image has more than {maxv} channels, only first {maxv} will be shown. Shape is {x.shape}')
        x = x[:,:,:maxv]

    return x