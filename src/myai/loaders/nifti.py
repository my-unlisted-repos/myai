"""nifti"""
import nibabel as nib
import SimpleITK as sitk
# import ants
import numpy as np
import torch

def niiread(path:str) -> np.ndarray:
    return np.asanyarray(nib.load(path).dataobj) # type:ignore

def niiread_affine(path:str) -> np.ndarray:
    return nib.load(path).affine # type:ignore

def niiwrite(path, arr:np.ndarray | torch.Tensor, affine:list|np.ndarray, dtype=None):
    if isinstance(arr, torch.Tensor): arr = arr.numpy()
    nib.save((nib.Nifti1Image(arr, affine, dtype=dtype)), path) # type:ignore


def niireadtensor(path:str):
    arr = niiread(path)
    if arr.dtype == np.uint16: arr = arr.astype(np.int32)
    return torch.from_numpy(arr)

def niiwrite_nib(path:str, arr:np.ndarray | torch.Tensor, affine:np.ndarray | None):
    if isinstance(arr, torch.Tensor): arr = arr.numpy()
    nib.save((nib.Nifti1Image(arr, affine)), path) # type:ignore

def niiwrite_sitk(path:str, arr:np.ndarray | torch.Tensor, clevel = 9):
    if clevel < 1: usec = False
    else: usec = True
    if isinstance(arr, torch.Tensor): arr = arr.numpy()
    sitk.WriteImage(sitk.GetImageFromArray(arr), path, useCompression=usec, compressionLevel=clevel) # type:ignore

def niiread_sitk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def niiread_nib(path):
    return np.asanyarray(nib.load(path).dataobj) # type:ignore

# def niiread_ants(path):
#     import ants
#     return ants.image_read(path, pixeltype='float').numpy()

