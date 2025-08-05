import os

import numpy as np
import torch
import torchvision.datasets
from torchvision.transforms import v2

from .base import DATASETS_ROOT, Dataset

class TVDataset(Dataset):
    def __init__(self, name, cls, root = DATASETS_ROOT):
        """name - name of dataset used for folder name in my datasets folder

        cls - class like torchvision.datasets.MNIST should be CLASSIFICATION."""
        super().__init__()

        self.root = os.path.join(root, name)
        if not os.path.exists(self.root): os.mkdir(self.root)
        self.name = name
        self.cls = cls

        data_path = os.path.join(root, f'{self.name}.npz')

        # load
        if os.path.exists(data_path):
            data = np.load(data_path)
            images = torch.as_tensor(data['images'], dtype=torch.float32)
            labels = torch.as_tensor(data['labels'], dtype=torch.int64)
            self.mean = torch.as_tensor(data['mean'], dtype=torch.float32)
            self.std = torch.as_tensor(data['std'], dtype=torch.float32)

        # download
        else:
            train = self.cls(root = self.root, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]), download=True)
            test = self.cls(root = self.root, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]), download=True, train=False)

            # stack
            images = torch.stack([i[0] for i in train] + [i[0] for i in test])
            labels = torch.tensor([i[1] for i in train] + [i[1] for i in test], dtype=torch.int64)

            # znormalize
            mean = images.mean((0, 2,3), keepdim=True)
            std = images.std((0, 2,3), keepdim=True)
            images -= mean
            images /= std
            np.savez_compressed(
                os.path.join(self.root, f"{self.name}.npz"),
                images=images.numpy(force=True),
                labels=labels.numpy(force=True),
                mean=mean.numpy(force=True),
                std=std.numpy(force=True),
            )

        self.add_samples_(zip(images, labels))


def get_mnist(root = DATASETS_ROOT):
    """70,000 images (1, 28, 28), entire dataset is znormalized. Each image has integer label 0 to 9.
    First 60,000 samples are commonly used as train set."""
    return TVDataset('MNIST', torchvision.datasets.MNIST, root)

def get_cifar10(root = DATASETS_ROOT):
    """60,000 (3, 32, 32) znormalized per channel, labels 0 to 9, first 50,000 for train set"""
    return TVDataset('CIFAR10', torchvision.datasets.CIFAR10, root)

def get_cifar100(root = DATASETS_ROOT):
    """CIFAR100"""
    return TVDataset('CIFAR100', torchvision.datasets.CIFAR100, root)

def get_fashionmnist(root = DATASETS_ROOT):
    """70,000 images (1, 28, 28), entire dataset is znormalized. Each image has integer label 0 to 9.
    last 10,000 images are test set."""
    return TVDataset('FashionMNIST', torchvision.datasets.FashionMNIST, root)
