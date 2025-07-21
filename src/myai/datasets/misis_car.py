import csv
import itertools
import os

import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from .base import DATASETS_ROOT, Dataset
from ..loaders.image import imreadtensor
from ..torch_tools import pad_to_shape

MEAN = torch.tensor([[
    [[107.6875]],
    [[108.5625]],
    [[110.3750]]
]])
STD = torch.tensor([[
    [[74.8125]],
    [[75.3750]],
    [[76.9375]]
]])

_resize = v2.Resize((240, 320), antialias=False)
def _preprocess(x):
    """makes sure (3, 480, 640) doesnt normalize!!!"""
    if x.shape[0] == 1: x = torch.cat([x,x,x])
    return _resize(pad_to_shape(x,(3, 480, 640)),)

def _image_to_float32(x):
    """since datset saved in float16 this converts images to flaot32?"""
    image, label = x
    return image.to(torch.float32, copy=False), label


ROOT = os.path.join(DATASETS_ROOT, "MDS-MISIS-DL Car classification")

class MISISCar(Dataset):
    """
    inputs: 15,561 images, original was 3×480×640 but I resized to 3×240×320 and saved in float16 so that it fits into memory.
    Loader is set to convert to float32. Entire dataset is z-normalized per-channel.

    targets: 10 classes, each image has int64 label 0 to 9.
    """
    def __init__(self, root=ROOT, device=None):
        super().__init__()
        self.root = root

        data_path = os.path.join(root, 'train.npz')
        if os.path.exists(data_path):
            data = np.load(data_path)
            images = torch.as_tensor(data['images'], dtype=torch.float16)
            labels = torch.as_tensor(data['labels'], dtype=torch.int64)

        else:

            with open(os.path.join(ROOT, "train.csv"), 'r', encoding = 'utf8') as f:
                items = list(csv.reader(f))[1:]

            # have to use float16 and resize by 50% to make it fit into memory
            image_paths = [os.path.join(ROOT, "train", f"{i[1]}", f"{i[0]}") for i in items]
            images = [_preprocess(imreadtensor(p)) for p in image_paths]

            # stack
            images = torch.stack(images)
            labels = torch.tensor([int(i[1]) for i in items], dtype=torch.int64)

            # znormalize
            images -= MEAN
            images /= STD

            # save
            np.savez_compressed(data_path, images=images.numpy(force=True), labels=labels.numpy(force=True))


        # add samples
        self.add_samples_(zip(images, labels), loader=_image_to_float32)


    def preprocess(self, inputs, device=None):
        from ..transforms import totensor, to_3HW
        inputs = torch.stack([_preprocess(to_3HW(totensor(i, device=device, dtype=torch.float32))) for i in inputs])
        inputs -= MEAN
        inputs /= STD
        return inputs

    def make_val_submission(self, fname, model, batch_size = 32):
        """make a submission csv"""
        submission = "Id,Category\n"
        for ids in itertools.batched(tqdm(os.listdir(os.path.join(self.root, "test"))), batch_size):
            inputs = self.preprocess([os.path.join(self.root, "test", f) for f in ids])

            outputs = model(inputs)
            submission += '\n'.join([f'{i},{int(o)}' for i, o in zip(ids, outputs.argmax(1).detach().cpu())])
            submission += '\n'

        with open(fname, 'w', encoding = 'utf8') as f:
            f.write(submission[:-1])
