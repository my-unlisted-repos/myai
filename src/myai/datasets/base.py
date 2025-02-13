"""This used to be a class now its just a template for a dataset."""
import os
import typing

import numpy as np
import torch

DATASETS_ROOT = r'/var/mnt/hdd/Datasets'
root: str = '???'


def _make(*args, **kwargs) -> torch.Tensor | list[torch.Tensor] | typing.Any:
    """download or create the dataset and stack it into arrays, return the arrays."""
    raise NotImplementedError

def _save(*args, **kwargs):
    """save dataset arrays to disk for fast loading."""
    images, labels = _make()
    np.savez_compressed(os.path.join(root, 'train.npz'), images=images, labels = labels)

def get(*args, **kwargs):
    """load dataset saved by `_save`."""
    raise NotImplementedError

def make_val_submission(outfile, model, *args, **kwargs):
    """make a submission file"""
    raise NotImplementedError

def inference(self, model, input, *args, **kwargs):
    """run inference on some file, applies same transforms to it as `_make` / `get`."""
    raise NotImplementedError