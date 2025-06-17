"""35,551 images 3×80×60, entire dataset is znormalized per channel.
44 classes, each image has integer label 0 to 43.
Extremely imbalanced, some classes have 10000 samples and some have a single sample."""
import csv
import itertools
import os

import numpy as np
import torch
from tqdm import tqdm

from .base import DATASETS_ROOT
from ..data import DS
from ..loaders.image import imreadtensor
from ..torch_tools import pad_to_shape

# WRONG
MEAN = torch.tensor([[
    [[217.7353]],
    [[213.3106]],
    [[211.7530]]
]])
STD = torch.tensor([[
    [[68.3273]],
    [[71.5364]],
    [[72.6093]]
]])

def _preprocess(x):
    """makes sure shape is (3, 80, 60)"""
    # images are different shape, and sometimes only one channel
    if x.shape[0] == 1: x = torch.cat((x,x,x))
    return pad_to_shape(x, (3, 80, 60), mode = 'min')


# class _BoolArtFashion(Dataset):
root = os.path.join(DATASETS_ROOT, "BoolArt-Image-Classification")

def _make():
    """decode all images and stack into a big array + labels array."""
    # read the labels
    with open(os.path.join(root, "train.csv"), 'r', encoding = 'utf8') as f:
        items = list(csv.reader(f))[1:]

    # load the jpegs
    ds = [(
        _preprocess(imreadtensor(os.path.join(root, "train_image", f'{i[0]}.jpg'))),
        i[1]
    ) for i in items]

    # stack
    images = torch.stack([i[0] for i in ds]).to(torch.float32)
    labels = torch.tensor([i[1] for i in ds], dtype=torch.int64)

    # znormalize
    images -= MEAN
    images /= STD

    return images, labels

def _save():
    """save dataset arrays to disk for fast loading."""
    images, labels = _make()
    np.savez_compressed(os.path.join(root, 'train.npz'), images=images, labels = labels)


def get() -> DS[tuple[torch.Tensor, int]]:
    """returns DS with float32 (3, 80, 80) images and int64 labels 0 to 43."""
    data = np.load(os.path.join(root, 'train.npz'))
    images = data['images']
    labels = data['labels']
    ds = DS()
    ds.add_samples_([(torch.from_numpy(i), torch.tensor(l, dtype=torch.int64)) for i, l in zip(images, labels)])
    return ds


def make_val_submission(outfile, model, batch_size = 32):
    """make a submission csv"""
    submission = "id,predict\n"
    for ids in itertools.batched(tqdm(os.listdir(os.path.join(root, "test_image"))), batch_size):
        inputs = torch.stack([_preprocess(imreadtensor(os.path.join(root, "test_image", f), dtype=torch.float32)) for f in ids])
        inputs -= MEAN
        inputs /= STD

        outputs = model(inputs)
        submission += '\n'.join([f'{i.replace(".jpg", "")},{int(o)}' for i, o in zip(ids, outputs.argmax(1).detach().cpu())])
        submission += '\n'

    with open(outfile, 'w', encoding = 'utf8') as f:
        f.write(submission[:-1])

