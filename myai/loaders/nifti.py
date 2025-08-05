"""nifti"""
import importlib.util
from typing import Any, cast, TYPE_CHECKING, TypeGuard
import os

import numpy as np
import torch

if TYPE_CHECKING:
    from SimpleITK import Image as sitk_Image

if importlib.util.find_spec('SimpleITK') is not None: import SimpleITK as sitk # pylint: disable=import-error # type:ignore
else: sitk = None


if importlib.util.find_spec('nibabel') is not None: import nibabel as nib # pylint: disable=import-error # type:ignore
else: nib = None

if importlib.util.find_spec('ants') is not None: import ants # pylint: disable=import-error # type:ignore
else: ants = None

# ----------------------------------- main ----------------------------------- #
def niiread(path:str) -> np.ndarray:
    return _niiread_sitk(path)

def niiwrite(path, arr:"np.ndarray | torch.Tensor | sitk_Image", reference: "sitk_Image | None" = None, clevel = 9):
    return _niiwrite_sitk(path, arr, reference, clevel)

def niireadtensor(path:str):
    arr = niiread(path)
    if arr.dtype == np.uint16: arr = arr.astype(np.int32)
    return torch.from_numpy(arr)

# --------------------------------- SimpleITK -------------------------------- #

def _niiread_sitk(path):
    if sitk is None: raise ModuleNotFoundError("SimpleITK is not installed")
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def _niiwrite_sitk(path:str, arr:"np.ndarray | torch.Tensor | sitk_Image", reference: "sitk_Image | None" = None, clevel = 9):
    if sitk is None: raise ModuleNotFoundError("SimpleITK is not installed")
    if clevel < 1: usec = False
    else: usec = True
    if isinstance(arr, torch.Tensor): arr = arr.numpy()
    if isinstance(arr, np.ndarray): arr = sitk.GetImageFromArray(arr)
    assert isinstance(arr, sitk.Image)
    if reference is not None: arr.CopyInformation(reference)
    sitk.WriteImage(arr, path, useCompression=usec, compressionLevel=clevel)

# ---------------------------------- nibabel --------------------------------- #
def _niiread_nib(path):
    return np.asanyarray(nib.load(path).dataobj) # type:ignore

def _niiwrite_nib(path:str, arr:np.ndarray | torch.Tensor, affine:np.ndarray | None):
    if isinstance(arr, torch.Tensor): arr = arr.numpy()
    nib.save((nib.Nifti1Image(arr, affine)), path) # type:ignore

# ----------------------------------- ants ----------------------------------- #
def _niiread_ants(path):
    return ants.image_read(path, pixeltype='float').numpy() # type:ignore




def is_sitk(x) -> "TypeGuard[sitk_Image]":
    if sitk is None: return False
    return isinstance(x, sitk.Image)


def sitk_to_numpy(x: "sitk_Image") -> np.ndarray:
    assert sitk is not None
    return sitk.GetArrayFromImage(x)