import pickle
import importlib.util
import os
from typing import Any, cast, TYPE_CHECKING
from collections.abc import Mapping

import numpy as np
import torch

from ..python_tools import path_zoom, get_first_file
from .csv import csvread, csvwrite
from .image import imread, imreadtensor
from .text import txtread, txtwrite
from .nifti import niiread, niiwrite, is_sitk, sitk_to_numpy

if TYPE_CHECKING:
    from SimpleITK import Image as sitk_Image

def _is_installed(module: str):
    return importlib.util.find_spec(module) is not None

def _not_installed(msg):
    def fn(*args, **kargs) -> Any:
        raise ModuleNotFoundError(msg)
    return fn


if _is_installed('pedalboard'): from .audio import audioread, audiowrite
else: audioread = audiowrite = _not_installed("pedalboard is not installed")

if _is_installed('bloscpack'): from .blosc import bloscread, bloscwrite
else: bloscread = bloscwrite = _not_installed("bloscpack is not installed")

if _is_installed('pydicom'): from .dicom import dcmread, dcmread_folder, is_dicom
else: dcmread = dcmread_folder = is_dicom = _not_installed("pydicom is not installed")

if _is_installed('SimpleITK'): import SimpleITK as sitk
else: sitk = None

if _is_installed("ruamel.yaml"): from .yaml import yamlread, yamlwrite
else: yamlread = yamlwrite = _not_installed("ruamel.yaml is not installed")

def _extract_int_from_str(s: str):
    return int(''.join(c for c in s if c.isnumeric()))

def read(path: str) -> np.ndarray:
    # ---------------------------------- folder ---------------------------------- #
    if os.path.isdir(path):
        path = path_zoom(path)

        if os.path.isdir(path):
            sample = get_first_file(path)
            if is_dicom(sample):
                return dcmread_folder(path)

            files = [os.path.join(path, f) for f in sorted(os.listdir(path), key = _extract_int_from_str)]
            return np.stack([read(f) for f in files])

    xl = path.lower()
    # ----------------------------------- nifti ---------------------------------- #
    if xl.endswith(('nii', 'nii.gz')):
        return niiread(path)

    # ----------------------------------- audio ---------------------------------- #
    if xl.endswith(('mp3', 'flac', 'wav', 'ogg', 'aac')):
        return audioread(path)[0]

    # ----------------------------------- numpy ---------------------------------- #
    if xl.endswith(('npy', 'npz')):
        arr = np.load(path)
        if isinstance(arr, Mapping):
            if len(arr) > 1: raise ValueError(f"{xl} has more than 1 key: {list(arr.keys())}, idk what to load!")
            return arr[next(iter(arr.keys()))]

        from ..transforms import tonumpy
        return tonumpy(arr)

    # ---------------------------------- pickle ---------------------------------- #
    if xl.endswith('pkl'):
        with open(path, 'rb') as f:
            arr = pickle.load(f)

        from ..transforms import tonumpy
        return tonumpy(arr)

    return imread(path)


def tositk(x) -> "sitk_Image":
    assert sitk is not None
    if isinstance(x, str):

        if os.path.isdir(x):
            import mrid
            return mrid.tositk(x)

        x = read(x)

    if isinstance(x, sitk.Image): return x
    if isinstance(x, np.ndarray): return sitk.GetImageFromArray(x)
    if isinstance(x, torch.Tensor): return sitk.GetImageFromArray(x.detach().cpu().numpy())
    return sitk.GetImageFromArray(np.asarray(x))