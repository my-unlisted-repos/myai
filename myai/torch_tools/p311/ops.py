import typing as T

import numpy as np
import torch

from .pad_ import pad
def center_of_mass(feature:torch.Tensor):
    '''
    adapted to pytorch from
    https://github.com/tym002/tensorflow_compute_center_of_mass/blob/main/compute_center_mass.py

    COM computes the center of mass of the input 4D or 5D image
    To use COM in a tensorflow model, use layers.Lambda
    Arguments:
        feature: input image of 5D tensor with format [batch,channel,x,y,z]
                    or 4D tensor with format [batch,channel,x,y]
        nx,ny,nz: dimensions of the input image, if using 4D tensor, nz = None
    '''
    feature = feature.moveaxis(1, -1) # move channels last

    if feature.ndim == 3: nx, ny, nz = feature.shape
    elif feature.ndim == 2: nx, ny = feature.shape
    else: raise NotImplementedError
    map1 = feature.unsqueeze(0).unsqueeze(-1)
    n_dim = map1.ndim

    if n_dim == 5:
        x = torch.sum(map1, dim =(2,3))
    else:
        x = torch.sum(map1, dim = 2)

    r1 = torch.arange(0,nx, dtype = torch.float32)
    r1 = torch.reshape(r1, (1,nx,1))

    x_product = x*r1
    x_weight_sum = torch.sum(x_product,dim = 1,keepdim=True)+0.00001
    x_sum = torch.sum(x,dim = 1,keepdim=True)+0.00001
    cm_x = torch.divide(x_weight_sum,x_sum)

    if n_dim == 5:
        y = torch.sum(map1, dim =(1,3))
    else:
        y = torch.sum(map1, dim = 1)

    r2 = torch.arange(0,ny, dtype = torch.float32)
    r2 = torch.reshape(r2, (1,ny,1))

    y_product = y*r2
    y_weight_sum = torch.sum(y_product,dim = 1,keepdim=True)+0.00001
    y_sum = torch.sum(y,dim = 1,keepdim=True)+0.00001
    cm_y = torch.divide(y_weight_sum,y_sum)

    if n_dim == 5:
        z = torch.sum(map1, dim =(1,2))

        r3 = torch.arange(0,nz, dtype = torch.float32) # type:ignore # pylint:disable=E0606
        r3 = torch.reshape(r3, (1,nz,1)) # type:ignore

        z_product = z*r3
        z_weight_sum = torch.sum(z_product,dim = 1,keepdim=True)+0.00001
        z_sum = torch.sum(z,dim = 1,keepdim=True)+0.00001
        cm_z = torch.divide(z_weight_sum,z_sum)

        center_mass = torch.concat([cm_x,cm_y,cm_z],dim=1)
    else:
        center_mass = torch.concat([cm_x,cm_y],dim=1)

    return center_mass[0].squeeze(1)

def binary_erode3d(tensor: torch.Tensor, n: int = 1, padding_mode = 'replicate'):
    """
    Erodes a 3D binary tensor.
    """
    if n > 1: tensor = binary_erode3d(tensor, n-1)
    kernel = torch.tensor([[[[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]]]], dtype=tensor.dtype, device=tensor.device)
    padded_input = torch.nn.functional.pad(tensor.unsqueeze(0), (1,1,1,1,1,1), mode = 'replicate')
    convolved = torch.nn.functional.conv3d(input = padded_input, weight = kernel, padding=0) # pylint:disable=E1102
    return torch.where(convolved==7, 1, 0)[0]

def one_hot_mask(mask: torch.Tensor, num_classes:int) -> torch.Tensor:
    """Takes a mask `(*)` and one-hot encodes into `(C, *)`"""
    return torch.nn.functional.one_hot(mask.to(torch.int64), num_classes).moveaxis(-1, 0) # type:ignore # pylint:disable=E1102

def batched_one_hot_mask(mask: torch.Tensor, num_classes:int) -> torch.Tensor:
    """Takes a batch of masks `(B, *)` and one-hot encodes into `(B, C, *)`."""
    return torch.nn.functional.one_hot(mask.to(torch.int64), num_classes).moveaxis(-1, 1) # type:ignore # pylint:disable=E1102

def raw_preds_to_one_hot(raw: torch.Tensor) -> torch.Tensor:
    """Takes raw model predictions in `(C, *)` format and turns into one-hot encoding in `(C, *)` format."""
    mask = torch.argmax(raw, dim=0)
    return one_hot_mask(mask, raw.shape[0])

def batched_raw_preds_to_one_hot(raw: torch.Tensor) -> torch.Tensor:
    """Takes a batch of raw model predictions in `(B, C, *)` format and turns into one-hot encoding in `(B, C, *)` format."""
    mask = torch.argmax(raw, dim=1)
    return batched_one_hot_mask(mask, raw.shape[1])


@T.overload
def stepchunk(vec: torch.Tensor, chunks: int, maxlength: int | None = None) -> list[torch.Tensor]: ...
@T.overload
def stepchunk(vec: np.ndarray, chunks: int, maxlength: int | None = None) -> list[np.ndarray]: ...
def stepchunk(vec:torch.Tensor|np.ndarray, chunks:int, maxlength:int | None=None) -> list[torch.Tensor] | list[np.ndarray]:
    """Chunk a vector, but using steps (e.g. first chunk can be 0,4,8,12,16, second - 1,5,9,13,17, etc)"""
    maxlength = maxlength or vec.shape[0]
    return [vec[i : i+maxlength : chunks] for i in range(chunks)] # type:ignore
