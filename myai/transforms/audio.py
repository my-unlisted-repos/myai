import typing as T
import numpy as np, torch

def to_duration(x:torch.Tensor, sr, sec):
    """X must be a 2d (channel, samples) tensor. This either cuts the audio or extends with zeroes."""

    # cut to sec
    x = x[:, :int(sr*sec)]

    # extend to sec
    if x.shape[1] < int(sr*sec):
        x = torch.cat([x, torch.zeros(x.shape[0], int(sr*sec - x.shape[1]))], dim=1)
    return x

def normalize(x:torch.Tensor):
    """Normalizes the audio to [-1, 1]"""
    return x / x.abs().max()

@T.overload
def dcoffset_fix(x: np.ndarray) -> np.ndarray: ...
@T.overload
def dcoffset_fix(x: torch.Tensor) -> torch.Tensor: ...
def dcoffset_fix(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Fixes DC offset."""
    return x - x.mean()

@T.overload
def cut(x: np.ndarray, sr:int, start_sec: float | None = None, end_sec: float | None = None) -> np.ndarray: ...
@T.overload
def cut(x: torch.Tensor, sr:int, start_sec: float | None = None, end_sec: float | None = None) -> torch.Tensor: ...
def cut(x: np.ndarray | torch.Tensor, sr:int, start_sec: float | None = None, end_sec: float | None = None) -> np.ndarray | torch.Tensor:
    """X must be a 2d (channel, samples) array. This either cuts the audio or extends with zeroes."""
    if start_sec is None:
        start_sec = 0
    if end_sec is None:
        end_sec = x.shape[1] / sr
    return x[:, int(sr*start_sec):int(sr*end_sec)]