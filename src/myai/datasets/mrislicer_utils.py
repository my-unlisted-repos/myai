"""my mri utils but not all of them inference ones in the contract lib plus some in mrid also i should add all new stuff to mrid"""
import typing as T
from collections import abc
from functools import partial
from operator import call

import joblib
import torch
from mrid import totensor
from mrid.training.mri_slicer import MRISlicer, randcrop2dt

from ..data import DS
from ..plt_tools import imshow
from ..python_tools import reduce_dim
from ..torch_tools import crop_around, one_hot_mask, overlay_segmentation


def load_old_mrislicer_dataset(
    path, num_classes: int, around: int, any_prob: float = 0.1, warn_empty: bool = True
) -> list[MRISlicer]:
    """Load dataset created with old glio MRISlicer"""
    old_format = joblib.load(path)
    dataset = []
    for sample in old_format:
        dataset.append(
            MRISlicer(
                sample.tensor,
                sample.seg,
                num_classes=num_classes,
                around=around,
                any_prob=any_prob,
                warn_empty=warn_empty,
            )
        )

    return dataset

def load_mrislicer_dataset(path) -> list[MRISlicer]:
    """Load dataset created with new mrid MRISlicer (just typed joblib.load)"""
    return joblib.load(path)

def get_seg_slices(dataset: list[MRISlicer] | str, around: int, any_prob: float = 0.5) -> list[abc.Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    """Returns all slices in a dataset that contain segmentation + `any_prob` * 100 % objects that return a random slice."""
    # load dataset
    if isinstance(dataset, str): dataset = load_mrislicer_dataset(dataset)

    # set settings that differ each time
    for slicer in dataset: slicer.set_settings(around = around, any_prob=any_prob)

    # get all segmentation slices as callables
    ds = reduce_dim([i.get_all_seg_slice_callables() for i in dataset])

    # get random slice getter callables
    random_slices = reduce_dim([i.get_anyp_random_slice_callables() for i in dataset])

    # return the list
    ds.extend(random_slices)
    return ds

def get_all_slices(dataset: list[MRISlicer] | str, around: int) -> list[abc.Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    """Returns all slices in a dataset
    Note that when using this for test dataset,
    it may make test loss better than it is because most images have the easy to segment nothing on them."""
    if isinstance(dataset, str): dataset = load_mrislicer_dataset(dataset)
    for slicer in dataset: slicer.set_settings(around = around)
    ds = reduce_dim([i.get_all_slice_callables() for i in dataset])
    return ds

# mrid version better
# def randcrop(x: tuple[torch.Tensor, torch.Tensor], num_classes: int, size = (96,96),):
#     """randomly crop x which is tuple of `(image, mask)` where image is `(C, H, W)` and mask is `(H, W)`
#     (mask will be one hot encoded)."""
#     if x[0].shape[1] == size[0] and x[0].shape[2] == size[1]: return x
#     if x[0].shape[1] < size[0] or x[0].shape[2] < size[1]: raise ValueError("Image is too small to crop to size")
#     #print(x[0].shape)
#     startx = random.randint(0, (x[0].shape[1] - size[0]) - 1)
#     starty = random.randint(0, (x[0].shape[2] - size[1]) - 1)
#     return (
#         x[0][:, startx:startx+size[0], starty:starty+size[1]].to(torch.float32),
#         one_hot_mask(x[1][startx:startx+size[0], starty:starty+size[1]], num_classes)
#     )


def _one_hot_x_mask(x: tuple[torch.Tensor, torch.Tensor], num_classes: int):
    return x[0], one_hot_mask(x[1], num_classes)

def _unsqueeze_x_mask(x: tuple[torch.Tensor, torch.Tensor]):
    return x[0], x[1].unsqueeze(0)

def make_ds(dataset: list[MRISlicer] | str, num_classes: int, around: int, any_prob: float = 0.5, window_size = (96,96)) -> DS:
    """Use for both train and test datasets. Since any_prob 0.5 works better anyway.
    Each sample in returned ds is a tuple with `(C, *window_size)` image
    and `(cls, *window_size)` one-hot encoded mask (or binary if num classes 1)."""
    ds = DS()
    slices = get_seg_slices(dataset, around = around, any_prob = any_prob)
    if num_classes == 1: transforms = [partial(randcrop2dt, size = window_size), _unsqueeze_x_mask]
    else: transforms = [partial(randcrop2dt, size = window_size), partial(_one_hot_x_mask, num_classes = num_classes)]
    ds.add_samples_(slices, transform=transforms, call=True)
    return ds


class ReferenceSlice:
    """Holds a reference image and segmentation to test model on."""
    def __init__(self, image, seg, slice: int, coord: tuple[int,int], axis = 0, flip_dims = None):
        """This.

        :param image: `(C, D, H, W)` or `(D, H, W)`.
        :param seg: `(D, H, W)` or one hot encoded `(C, D, H, W)`
        :param slice: slice on the transformed image
        :param coord: center coord on the slices
        """
        self.image = totensor(image)
        if self.image.ndim == 3: self.image = self.image.unsqueeze(0)

        self.seg = totensor(seg)
        if self.seg.ndim == 4: self.seg = self.seg.argmax(0)

        if axis != 0:
            self.image = self.image.moveaxis(axis+1, 1)
            self.seg = self.seg.moveaxis(axis, 0)

        if flip_dims is not None:
            if isinstance(flip_dims, int): flip_dims = [flip_dims,]
            self.image = torch.flip(self.image, [i+1 for i in flip_dims])
            self.seg = torch.flip(self.seg, flip_dims)

        self.slice = slice
        self.coord = coord

    def show(self, size = 64, mod = 0):
        if isinstance(size, int): size = (size, size)
        image_slice = crop_around(self.image[mod,self.slice], self.coord, size)
        if self.seg is not None:
            seg_slice = crop_around(self.seg[self.slice], self.coord, size)
            image_slice = overlay_segmentation(image_slice, seg_slice, 0.25)

        imshow(image_slice).show()

    def show_full(self, mod = 0):
        image_slice = self.image[mod, self.slice]
        if self.seg is not None:
            seg_slice = self.seg[self.slice]
            image_slice = overlay_segmentation(image_slice, seg_slice, 0.25)

        imshow(image_slice).show()

    def get(self, size: int | tuple[int, int], around: int):
        """Returns `(*, *size)` tensor and `(*size)` seg not one hot encoded.
        Where all modalities will be sequential, e.g. first 7 slices of t1c, then 7 slices of t1n, etc."""
        if isinstance(size, int): size = (size, size)
        if around != 0:
            image_slice = crop_around(self.image[:,self.slice-around:self.slice+around+1], self.coord, size).flatten(0,1)
        else:
            image_slice = crop_around(self.image[:,self.slice], self.coord, size)
        if self.seg is not None:
            seg_slice = crop_around(self.seg[self.slice], self.coord, size)
        else:
            seg_slice = T.cast(torch.Tensor, None)
        return image_slice, seg_slice

    def save(self, path: str):
        joblib.dump(self, path, compress=5)

    @classmethod
    def load(cls, path: str) -> "ReferenceSlice":
        return joblib.load(path)