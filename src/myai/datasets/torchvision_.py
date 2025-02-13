import os

import numpy as np
import torch
import torchvision.datasets
from torchvision.transforms import v2

from ..data import DS
from .base import DATASETS_ROOT


class _TorchvisionClassificationDataset:
    """same as dataset modules but in a class so that it can be created from any tv dataset"""
    def __init__(self, name, cls, root = DATASETS_ROOT):
        """name - name of dataset used for folder name in my datasets folder

        cls - class like torchvision.datasets.MNIST should be CLASSIFICATION."""
        self.root = os.path.join(root, name)
        self.name = name
        self.cls = cls

    def _make(self):
        """downloads stacks calculates mean std and znormalizes and returns images and labels"""
        # download
        train = self.cls(root = self.root, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]), download=True)
        test = self.cls(root = self.root, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]), download=True, train=False)

        # stack
        images = torch.stack([i[0] for i in train] + [i[0] for i in test])
        labels = torch.tensor([i[1] for i in train] + [i[1] for i in test], dtype=torch.int64)

        # znormalize
        images -= images.mean((0, 2,3), keepdim=True)
        images /= images.std((0, 2,3), keepdim=True)

        return images, labels

    def _save(self):
        """save znormalized dataset arrays to disk for fast loading."""
        if not os.path.exists(self.root):
            try:
                os.mkdir(self.root)
            except Exception as e:
                print(f'hey idiot you are not me so your not allowed to use this😡:\n{e!r}')

        images, labels = self._make()
        np.savez_compressed(os.path.join(self.root, f'{self.name}.npz'), images=images, labels = labels)

    def get(self) -> DS[tuple[torch.Tensor, int]]:
        """returns DS with float32 images and int64 labels. this should work assuming i ran `_save` first on this
        so the dataset is in my datasets folder which is self.name."""
        ds_path = os.path.join(self.root, f'{self.name}.npz')
        if not os.path.exists(ds_path):
            print(f"`{ds_path}` doesn't exist; downloading and creating dataset...")
            self._save()

        data = np.load(os.path.join(self.root, f'{self.name}.npz'))
        images = data['images']
        labels = data['labels']
        ds = DS()
        ds.add_samples_([(torch.from_numpy(i), torch.tensor(l, dtype=torch.int64)) for i, l in zip(images, labels)])
        return ds


MNIST = _TorchvisionClassificationDataset('MNIST', torchvision.datasets.MNIST)
"""70,000 images (1, 28, 28), entire dataset is znormalized. Each image has integer label 0 to 9.
First 60,000 samples are commonly used as train set."""
CIFAR10 = _TorchvisionClassificationDataset('CIFAR10', torchvision.datasets.CIFAR10)
"""60,000 (3, 32, 32) znormalized per channel, labels 0 to 9, first 50,000 for train set"""
FashionMNIST = _TorchvisionClassificationDataset('FashionMNIST', torchvision.datasets.FashionMNIST)
"""70,000 images (1, 28, 28), entire dataset is znormalized. Each image has integer label 0 to 9.
last 10,000 images are test set."""