from collections.abc import Callable, Sequence
from typing import Literal

import torch

from .block_base import Block, BlockType
from .containers import ensure_block
from ..func import ensure_module
from ...python_tools import maybe_compose, Composable

AggregateModes = Literal['cat', 'sum', 'prod', 'min', 'max']

class _AggeregateModule(torch.nn.Module):
    def __init__(self, mode: AggregateModes, in_modules: Sequence[Composable | None] | None, out_module: Composable | None):
        super().__init__()
        self.in_modules = [ensure_module(m) if m is not None else m for m in in_modules] if in_modules is not None else None
        self.out_module = ensure_module(out_module) if out_module is not None else None
        self.mode: AggregateModes = mode

    def forward(self, *args):
        if self.in_modules is not None: args = [m(a) if m is not None else a for a, m in zip(args, self.in_modules)]

        if self.mode == 'cat': res = torch.cat(args, 1)
        elif self.mode == 'sum': res = torch.sum(torch.stack(args, 0), 0)
        elif self.mode == 'prod': res = torch.prod(torch.stack(args, 0), 0)
        elif self.mode == 'min': res = torch.amin(torch.stack(args, 0), 0)
        elif self.mode == 'max': res = torch.amax(torch.stack(args, 0), 0)
        else: raise NotImplementedError(self.mode)

        if self.out_module is not None: return self.out_module(res)
        return res



class Aggregate(Block):
    def __init__(self, mode: AggregateModes = 'cat', in_blocks: BlockType | Sequence[BlockType | None] | None = None, out_block: BlockType | None = None):
        super().__init__()
        self.mode: AggregateModes = mode
        self.in_blocks = in_blocks
        self.out_block = out_block

    def _make(self, in_channels, out_channels, channel_scale, in_size, out_size, size_scale, ndim, channel_dim):
        assert in_channels is None

        # make input blocks
        in_blocks = self.in_blocks if isinstance(self.in_blocks, Sequence) else [self.in_blocks for _ in self.agg_channels]
        in_modules = [ensure_block(b).new(in_channels=c, out_channels=c, in_size = in_size, out_size = in_size, ndim=ndim, channel_dim=channel_dim) if b is not None else None for b,c in zip(in_blocks, self.agg_channels)]

        # determine how many channels we have after aggregating
        if self.mode == 'cat': skip_out_channels = sum(self.agg_channels)
        else:
            if len(set(self.agg_channels)) != 1:
                raise ValueError(f'all values in in_channels must be the same if mode is not "cat", got {in_channels = }')
            skip_out_channels = self.agg_channels[0]

        # if out_channels is not None there must be an out block to set out_channels to the value unless it is already correct
        if self.out_block is None:
            if out_channels is not None:
                if out_channels != skip_out_channels:
                    raise ValueError(f"out_channels != skip_out_channels, Skip can't change channels without out_block. {out_channels = }, {skip_out_channels = }")
            if size_scale != 1: raise ValueError(f"Skip can't change size without an out_block, got {size_scale = }")

        # make out block
        if self.out_block is None: out_module = None
        else: out_module = ensure_block(self.out_block).new(skip_out_channels, out_channels, in_size=in_size, out_size=out_size, size_scale=size_scale, ndim=ndim, channel_dim = channel_dim)

        return _AggeregateModule(mode = self.mode, in_modules = in_modules,out_module = out_module)


    def new(
        self,
        agg_channels: Sequence[int],
        out_channels: int | None = None,
        in_size: int | Sequence[int] | None = None,
        out_size: int | Sequence[int] | None = None,
        size_scale: float | None = None,
        ndim: int = 2,
        channel_dim: int = 1,
    ):
        self.agg_channels = agg_channels
        return super().new(
            in_channels=None,
            out_channels=out_channels,
            channel_scale=None,
            in_size = in_size,
            out_size = out_size,
            size_scale = size_scale,
            ndim = ndim,
            channel_dim = channel_dim,
        )