import csv
import itertools
import os

import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from .base import DATASETS_ROOT, Dataset
from ..loaders.image import imreadtensor
from ..python_tools import get_all_files
from ..torch_tools import pad_to_shape
from ..transforms import to_HW3, resize_to_contain, resize_to_fit


def preprocess(x:torch.Tensor, resize_fn, size: tuple[int,int]):
    """resizes images to contain size*size square and crop to that square and znormalize and make sure 4 channels sybau icl tc pmo"""
    if size[0] != size[1]: raise NotImplementedError(size)

    # make sure it (4, H, W)
    x = to_HW3(x, allow_4_channels=True).moveaxis(-1, 0)
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

ROOT = os.path.join(DATASETS_ROOT, "My Images")

class MyImages(Dataset):
    """
    inputs: about 1200 images 4×128×128 or other size. Each image is z-normalized separately, not per-channel.
    """

    def __init__(self, root=ROOT, size = (128, 128), add_ui_stuff=False, contain=True):
        super().__init__()
        self.size = size
        self.contain = contain

        folder = f'{size[0]}x{size[1]}'
        if add_ui_stuff: folder = f'{folder} w.ui'
        folder = f'{folder} {"contain" if contain else "fit"}'
        root = os.path.join(root, folder)
        if not os.path.exists(root): os.mkdir(root)
        self.root = root

        # load
        data_path = os.path.join(root, 'train.npz')
        if os.path.exists(data_path):
            data = np.load(data_path)
            images = torch.as_tensor(data['images'], dtype=torch.float32)

        # make
        else:
            resize_fn = resize_to_contain if contain else resize_to_fit

            files = get_all_files(r'/var/mnt/ssd/Files/Documents/Изображения', extensions=extensions)
            if add_ui_stuff: files.extend(get_all_files(r'/var/mnt/ssd/Files/.themes/Orchis-Indigo-Compact', extensions=extensions))

            dataset = []
            for f in tqdm(files):
                image = preprocess(imreadtensor(f), resize_fn = resize_fn, size = size)
                if image is not None:
                    dataset.append(image)

            images = torch.stack(dataset)
            np.savez_compressed(os.path.join(root, 'train.npz'), images=images.numpy(force=True))

        # add samples
        self.add_samples_(images)


    def preprocess(self, inputs):
        resize_fn = resize_to_contain if self.contain else resize_to_fit
        inputs = [preprocess(imreadtensor(i), resize_fn = resize_fn, size = self.size) for i in inputs]
        return torch.stack([i for i in inputs if i is not None])

