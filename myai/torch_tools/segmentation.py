import itertools
import typing as T

import numpy as np
import torch

from ..transforms import totensor

COLORS = {
    "red": (1, 0, 0),
    "r": (1, 0, 0),
    "green": (0, 1, 0),
    "g": (0, 1, 0),
    "blue": (0, 0, 1),
    "b": (0, 0, 1),
    "yellow": (1, 1, 0),
    "y": (1, 1, 0),
    "cyan": (0, 1, 1),
    "c": (0, 1, 1),
    "magenta": (1, 0, 1),
    "m": (1, 0, 1),
    "black": (0, 0, 0),
    "white": (1, 1, 1),
    "w": (1, 1, 1),
    "orange": (1, 0.5, 0),
    "o": (1, 0.5, 0),
    "purple": (0.5, 0, 0.5),
    "p": (0.5, 0, 0.5),
    "gray": (0.5, 0.5, 0.5),
    "lightblue": (0.2, 0.6, 1),
}

def overlay_segmentation(x: torch.Tensor | np.ndarray, seg:torch.Tensor | np.ndarray, alpha: float = 0.5, colors = None, bg_index: int | None = 0):
    """_summary_

    :param x: Tensor to put segmentation onto. Must be `(C, *)`. Will be converted to `(3, *)`.
    :param seg: Binarized (without C dimension) segmentation broadcastable into x.
    :param alpha: Low alpha means segmentation is more transparent, defaults to 0.5
    :param colors: List of tuples of three numbers from 0 to 1, color names, or None, defaults to None
    """
    x = totensor(x).clone().float()
    seg = totensor(seg).clone()

    n_classes = int(seg.max().detach().cpu()) + 1
    if bg_index is None:
        n_classes -= 1

    # generate colors if None
    if colors is None:
        colors = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]
        n = 2
        while len(colors) < n_classes:
            colors = list(itertools.product(np.linspace(0, 1, n), repeat=3))
            if (0., 0., 0.) in colors: colors.remove((0., 0., 0.)) # remove black
            if (1., 1., 1.) in colors: colors.remove((1., 1., 1.)) # remove white
            n += 1

    if isinstance(colors, str): colors = [colors]
    colors = [torch.tensor(COLORS[i.lower().strip()] if isinstance(i, str) else i, dtype=x.dtype, device=x.device) for i in colors]
    min = x.min(); max = x.max()
    if min != max: colors = [i * (max - min) + min for i in colors]

    if bg_index is not None:
        colors = list(colors)
        colors.insert(bg_index, T.cast(T.Never, None))


    if len(colors) < n_classes:
        raise ValueError(f"Not enough colors provided, need {n_classes} but got {len(colors)}")

    # make sure X is (3, *)
    if x.ndim == seg.ndim:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = torch.cat((x,x,x), 0)
    elif x.shape[0] == 2:
        x = torch.cat((x,x), 0)[:-1]
    elif x.shape[0] > 3:
        x = x[:3]

    for cls, c in zip(range(n_classes), colors):
        if cls == bg_index: continue

        mask = seg == cls
        color = c.unsqueeze(-1)
        x[:, mask] = x[:, mask] - ((x[:, mask] - color)) * alpha

    return x


def make_segmentation_overlay(seg:torch.Tensor | np.ndarray, colors = None, bg_index = 0, bg_color = (0,0,0)):
    """Takes (*) segmentation, returns (3, *) colored overlay."""
    seg = totensor(seg, device = torch.device('cpu'))
    x = torch.ones_like(seg)
    x = (torch.stack([x,x,x], -1) * torch.tensor(bg_color)).moveaxis(-1, 0)
    x = overlay_segmentation(x, seg, alpha = 1, colors=colors, bg_index=bg_index)
    return x