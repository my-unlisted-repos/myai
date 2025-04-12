"""nifti"""
import importlib.util

import mrid
import numpy as np
import SimpleITK as sitk
import torch

if importlib.util.find_spec('nibabel') is not None:
    import nibabel as nib # pylint: disable=import-error # type:ignore
else:
    nib = None

if importlib.util.find_spec('ants') is not None:
    import ants # pylint: disable=import-error # type:ignore
else:
    ants = None

# ----------------------------------- main ----------------------------------- #
def niiread(path:str) -> np.ndarray:
    return niiread_sitk(path)

def niiwrite(path, arr:np.ndarray | torch.Tensor | sitk.Image, reference: sitk.Image | None = None, clevel = 9):
    return niiwrite_sitk(path, arr, reference, clevel)

def niireadtensor(path:str):
    arr = niiread(path)
    if arr.dtype == np.uint16: arr = arr.astype(np.int32)
    return torch.from_numpy(arr)

# --------------------------------- SimpleITK -------------------------------- #

def niiread_sitk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def niiwrite_sitk(path:str, arr:np.ndarray | torch.Tensor | sitk.Image, reference: sitk.Image | None = None, clevel = 9):
    if clevel < 1: usec = False
    else: usec = True
    if isinstance(arr, torch.Tensor): arr = arr.numpy()
    if isinstance(arr, np.ndarray): arr = sitk.GetImageFromArray(arr)
    if reference is not None: arr.CopyInformation(reference)
    sitk.WriteImage(arr, path, useCompression=usec, compressionLevel=clevel)

# ---------------------------------- nibabel --------------------------------- #
def niiread_affine(path:str) -> np.ndarray:
    return nib.load(path).affine # type:ignore

def niiread_nib(path):
    return np.asanyarray(nib.load(path).dataobj) # type:ignore

def niiwrite_nib(path:str, arr:np.ndarray | torch.Tensor, affine:np.ndarray | None):
    if isinstance(arr, torch.Tensor): arr = arr.numpy()
    nib.save((nib.Nifti1Image(arr, affine)), path) # type:ignore

# ----------------------------------- ants ----------------------------------- #
def niiread_ants(path):
    return ants.image_read(path, pixeltype='float').numpy() # type:ignore

