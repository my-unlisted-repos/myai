import csv
import itertools
import os

import numpy as np
import torch
from tqdm import tqdm

from .base import DATASETS_ROOT, Dataset
from ..loaders.image import imreadtensor
from ..torch_tools import pad_to_shape

ROOT = os.path.join(DATASETS_ROOT, "BoolArt-Fashion")


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

class BoolArtFashion(Dataset):
    """
    inputs: 35,551 images 3×80×60. Entire dataset is z-normalized per-channel.

    targets: 44 classes, each image has integer label 0 to 43.

    Extremely imbalanced, some classes have 10000 samples and some have a single sample.
    """

    def __init__(self, path=ROOT, device=None):
        super().__init__()
        self.path = path

        # load
        data_path = os.path.join(path, 'train.npz')
        if os.path.exists(data_path):
            data = np.load(data_path)
            images = data['images']
            labels = data['labels']

        # make
        else:
            with open(os.path.join(path, "train.csv"), 'r', encoding = 'utf8') as f:
                items = list(csv.reader(f))[1:]

            # load the jpegs
            image_paths = [os.path.join(path, "train_image", f'{i[0]}.jpg') for i in items]
            images = [_preprocess(imreadtensor(p)) for p in image_paths]

            # stack
            images = torch.stack([i for i in images]).to(torch.float32)
            labels = torch.tensor([i[1] for i in items], dtype=torch.int64)

            # znormalize
            images -= MEAN
            images /= STD

            # save
            np.savez_compressed(os.path.join(path, 'train.npz'), images=images, labels=labels)

        # add samples
        samples = [(torch.from_numpy(i).to(device), torch.tensor(l, device=device, dtype=torch.int64)) for i, l in zip(images, labels)]
        self.add_samples_(samples)



    def submission(self, fname, model, batch_size = 32):
        """make a submission csv"""
        submission = "id,predict\n"
        for ids in itertools.batched(tqdm(os.listdir(os.path.join(self.path, "test_image"))), batch_size):
            inputs = torch.stack([_preprocess(imreadtensor(os.path.join(self.path, "test_image", f), dtype=torch.float32)) for f in ids])
            inputs -= MEAN
            inputs /= STD

            outputs = model(inputs)
            submission += '\n'.join([f'{i.replace(".jpg", "")},{int(o)}' for i, o in zip(ids, outputs.argmax(1).detach().cpu())])
            submission += '\n'

        with open(fname, 'w', encoding = 'utf8') as f:
            f.write(submission[:-1])

