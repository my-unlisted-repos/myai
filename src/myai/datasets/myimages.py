"""about 1200 (4, 128, 128). images only no labels. already znormalized and everything"""

import csv
import itertools
import os

import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from ..data import DS
from ..loaders.image import imreadtensor
from ..python_tools import get_all_files
from ..torch_tools import pad_to_shape
from ..transforms import force_hw3, resize_to_contain, resize_to_fit


def preprocess(x:torch.Tensor, resize_fn, size: tuple[int,int]):
    """resizes images to contain size*size square and crop to that square and znormalize and make sure 4 channels"""
    if size[0] != size[1]: raise NotImplementedError(size)

    # make sure it (4, H, W)
    x = force_hw3(x, allow_4_channels=True).moveaxis(-1, 0)
    if x.shape[0] == 3: x = torch.cat([x, torch.full_like(x[0], 255)[None]])

    x = resize_fn(x, size[0], antialias=False)
    x = pad_to_shape(x, (4,*size), crop=True)

    # znormalize (we znormalize each image because they are wildly different)
    x = x.to(torch.float32, copy=True)
    x -= x.mean()
    if x.std() == 0: return None
    x /= x.std()

    assert x.shape == (4, *size), x.shape
    return x

extensions = ['jpg', 'png', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'avif', 'jfif']

root = r"E:\datasets\My Images"

def _make(resize_fn = resize_to_contain, size = (128, 128), add_ui_stuff=False):
    """decode all images and stack into a big array can also add ui stuff to make it harder."""
    images = get_all_files(r'F:\Stuff\Images', extensions=extensions)
    if add_ui_stuff: images.extend(get_all_files(r'F:\Stuff 2\.themes\Orchis-Indigo-Compact', extensions=extensions))

    dataset = []
    for im in tqdm(images):
        image = preprocess(imreadtensor(im), resize_fn = resize_fn, size = size)
        if image is not None:
            dataset.append(image)

    return torch.stack(dataset)


def _save(name, resize_fn = resize_to_contain, size = (128, 128), add_homework=False):
    """save dataset array to disk for fast loading."""
    images = _make(resize_fn = resize_fn, size = size, add_ui_stuff = add_homework)
    np.savez_compressed(os.path.join(root, f'{name}.npz'), images=images)

def get(path) -> DS[torch.Tensor]:
    """returns DS with float32 (4, 128, 128) images"""
    data = np.load(path)
    images = torch.from_numpy(data['images']).to(torch.float32, copy=False)

    ds = DS()
    ds.add_samples_(images)
    return ds

def inference(model, input: str, resize_fn = resize_to_contain, size = (128, 128)):
    """run inference on new image file and returns (4, 128, 128)"""
    image = preprocess(imreadtensor(input), resize_fn = resize_fn, size = size)
    if image is None: raise ValueError
    return model(image.unsqueeze(0))[0].detach().cpu()

