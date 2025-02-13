"""dicom tools"""

import logging
import os
import numpy as np
import torch
import torchvision.transforms.v2
import pydicom


def dcmread(path) -> np.ndarray:
    """Reads a DICOM file and returns a np.ndarray.

    Args:
        path (_type_): path to a DICOM file.

    Returns:
        np.ndarray: _description_
    """
    return pydicom.dcmread(path).pixel_array

def dcmreadtensor(path, dtype=None) -> torch.Tensor:
    arr = dcmread(path)
    if arr.dtype == np.uint16: arr = arr.astype(np.int16)
    return torch.as_tensor(arr, dtype=dtype)


def dcmread_sorted_paths(paths:list[str]) -> np.ndarray:
    """Reads a sorted sequence of DICOM files and stacks them into a 3D np.ndarray. Files must already be sorted.

    Args:
        paths (_type_): list of DICOM paths.

    Returns:
        np.ndarray: _description_
    """
    return np.array([pydicom.dcmread(i).pixel_array for i in paths], copy=False)

def dcmreadtensor_sorted_paths(paths:list[str], dtype=None) -> torch.Tensor:
    """Reads a sorted sequence of DICOM files and stacks them into a 3D torch.Tensor. Files must already be sorted.

    Args:
        paths (_type_): _description_
        dtype (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    arr = dcmread_sorted_paths(paths)
    if arr.dtype == np.uint16: arr = arr.astype(np.int16)
    return torch.as_tensor(arr, dtype=dtype)


def dcmread_paths(paths:list[str]) -> np.ndarray:
    """Reads an unsorted sequence of DICOM files, sorts them by `InstanceNumber` and stacks into a np.ndarray.

    Args:
        paths (_type_): _description_

    Returns:
        np.ndarray: _description_
    """
    images = sorted([pydicom.dcmread(i) for i in paths], key = lambda x: x.InstanceNumber)# pyright:ignore[reportPossiblyUnboundVariable]
    return np.array([i.pixel_array for i in images], copy=False)

def dcmreadtensor_paths(paths:list[str], dtype=None) -> torch.Tensor:
    """Reads an unsorted sequence of DICOM files, sorts them by `InstanceNumber` and stacks into a torch.Tensor.

    Args:
        paths (list[str]): _description_
        dtype (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    arr = dcmread_paths(paths)
    if arr.dtype == np.uint16: arr = arr.astype(np.int16)
    return torch.as_tensor(arr, dtype=dtype)


def dcmread_folder(path) -> np.ndarray:
    """Reads all files in a folder, sorts them by `InstanceNumber` and stacks into a np.ndarray.

    Args:
        path (_type_): _description_

    Returns:
        np.ndarray: _description_
    """
    paths = [os.path.join(path, i) for i in os.listdir(path)]
    return dcmread_paths(paths)

def dcmreadtensor_folder(path, dtype=None) -> torch.Tensor:
    """Reads all files in a folder, sorts them by `InstanceNumber` and stacks into a torch.Tensor.

    Args:
        path (_type_): _description_
        dtype (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    arr = dcmread_folder(path)
    if arr.dtype == np.uint16: arr = arr.astype(np.int16)
    return torch.as_tensor(arr, dtype=dtype)

def affine2d(ds:pydicom.Dataset):
    """Code from https://stackoverflow.com/a/63478571/15673832.

    Convert DICOM to affine. Except it doesn't work.

    Args:
        ds (pydicom.Dataset): _description_

    Returns:
        _type_: _description_
    """
    F11, F21, F31 = ds.ImageOrientationPatient[3:]
    F12, F22, F32 = ds.ImageOrientationPatient[:3]

    dr, dc = ds.PixelSpacing
    Sx, Sy, Sz = ds.ImagePositionPatient

    return np.array(
        [
            [F11 * dr, F12 * dc, 0, Sx],
            [F21 * dr, F22 * dc, 0, Sy],
            [F31 * dr, F32 * dc, 0, Sz],
            [0, 0, 0, 1]
        ]
    )

def dcmread_affine(path) -> np.ndarray:
    ds = pydicom.dcmread(path)
    return affine2d(ds)