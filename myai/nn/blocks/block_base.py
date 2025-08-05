from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from ...python_tools import Composable



@dataclass
class Config:
    in_channels: int | None
    out_channels: int | None
    channel_scale: float
    in_size: int | Sequence[int] | None
    out_size: int | Sequence[int] | None
    size_scale: float | Sequence[float]
    ndim: int
    channel_dim: int

class Block(ABC):
    def __init__(self):
        self._last_config = cast(Config, None)

    @abstractmethod
    def _make(
        self,
        in_channels: int | None,
        out_channels: int | None,
        channel_scale: float,
        in_size: Sequence[int] | None,
        out_size: Sequence[int] | None,
        size_scale: float,
        ndim: int,
        channel_dim: int,
    ) -> torch.nn.Module:
        ...

    def new(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        channel_scale: float | None = None,
        in_size: int | Sequence[int] | None = None,
        out_size: int | Sequence[int] | None = None,
        size_scale: float | None = None,
        ndim: int = 2,
        channel_dim: int = 1,
    ) -> torch.nn.Module:

        # this guarantees channel_scale is always set
        if (in_channels is None or out_channels is None) and channel_scale is None: channel_scale = 1

        # channel scale from in and out channel
        if (channel_scale is None) and (in_channels is not None) and (out_channels is not None):
            channel_scale = out_channels / in_channels

        # out channels from in channels and scale
        if (out_channels is None) and (in_channels is not None) and (channel_scale is not None):
            out_channels = round(in_channels * channel_scale)

        if isinstance(in_size, int): in_size = [in_size]*ndim
        if isinstance(out_size, int): out_size = [out_size]*ndim

        # this guarantees size_scale is always set
        if (in_size is None or out_size is None) and size_scale is None: size_scale = 1

        # size scale from in and out size
        if (size_scale is None) and (in_size is not None) and (out_size is not None):
            if len(in_size) != len(out_size): raise ValueError(f"{in_size = } has different length from {out_size = }")
            size_scale_l = [o/i for o,i in zip(out_size, in_size)]
            if len(set(size_scale_l)) != 1: raise ValueError(size_scale_l)
            size_scale = size_scale_l[0]

        # out_size from in_size and scale
        if (out_size is None) and (in_size is not None) and (size_scale is not None):
            out_size = [round(i*size_scale) for i in in_size]

        # channels checks
        if (in_channels is not None) and (out_channels is not None) and (channel_scale is not None):
            calculated_scale = out_channels / in_channels
            if calculated_scale != channel_scale:
                raise ValueError(f"Inconsistent {out_channels = }, {in_channels = }, {out_channels / in_channels = }, {channel_scale = }")

        # size checks
        if (in_size is not None) and (out_size is not None) and (size_scale is not None):
            predicted_size_scale = [o/i for o,i in zip(out_size, in_size)]
            if len(set(predicted_size_scale)) > 1 or predicted_size_scale[0] != size_scale:
                raise ValueError(f"Inconsistent {predicted_size_scale = }, {size_scale = }. {in_size = }; {out_size = }")

        # call _make
        return self._make(
            in_channels = in_channels,
            out_channels = out_channels,
            channel_scale = channel_scale, # type:ignore
            in_size = in_size,
            out_size = out_size,
            size_scale = size_scale, # type:ignore
            ndim = ndim,
            channel_dim = channel_dim,
        )

BlockType = Block | Sequence[Block] | Composable