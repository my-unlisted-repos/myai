import numpy as np
import pedalboard
import torch

from ..transforms import tonumpy


def audioread(path) -> tuple[np.ndarray, int]:
    """Returns audio `(channes, samples)` in (-1, 1) range and sr"""
    with pedalboard.io.AudioFile(path, 'r') as f: # pylint:disable=E1129 # type:ignore
        audio = f.read(f.frames)
        sr = f.samplerate
    return audio, sr

def audioreadtensor(path) -> tuple[torch.Tensor, int]:
    """Returns audio `(channes, samples)` in (-1, 1) range and sr"""
    audio, sr = audioread(path)
    return torch.from_numpy(audio), sr


def audiowrite(file: str, arr, sr: int):
    """Write an audiofile. Please include extension in filename.

    Args:
        file (str): output file.
        arr (Any): array, can be channel first or channel last or 1d.
        sr (int): sample rate.
    """
    arr = tonumpy(arr)
    arr = arr.squeeze()
    if arr.ndim == 1: arr = arr[None, :]
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T

    if arr.ndim > 2: raise ValueError(arr.shape)

    with pedalboard.io.AudioFile(file, "w", samplerate=sr, num_channels=arr.shape[0]) as f: # pylint:disable=not-context-manager
        f.write(arr)
