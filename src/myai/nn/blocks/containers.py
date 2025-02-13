from collections.abc import Sequence, Iterable
import torch
from ..func import ensure_module
from .block_base import Block, BlockType
from ...python_tools import maybe_compose

__all__ = ["Sequential", "Lambda", "ensure_block"]
class Sequential(Block):
    def __init__(self, blocks: Sequence[BlockType], channel_idx: int, size_idx: int):
        """Sequential block

        Args:
            blocks (Sequence[Block]): blocks
            channel_idx (int): index of the block that changes number of channels.
            size_idx (int): index of the block that changes spatial size.
        """
        super().__init__()
        self.blocks = [ensure_block(b) for b in blocks]
        self.channel_idx = channel_idx
        self.size_idx = size_idx

    def _make(self, in_channels, out_channels, channel_scale, in_size, out_size, size_scale, ndim, channel_dim):
        modules = []

        channel_idx = list(range(len(self.blocks)))[self.channel_idx]
        size_idx = list(range(len(self.blocks)))[self.size_idx]

        for i,b in enumerate(self.blocks):
            out_ch = out_channels if i == channel_idx else in_channels
            ch_scale = channel_scale if i == channel_idx else 1

            out_s = out_size if i == size_idx else in_size
            s_scale = size_scale if i == size_idx else 1

            block = b.new(
                in_channels=in_channels,
                out_channels=out_ch,
                channel_scale=ch_scale,
                in_size=in_size,
                out_size=out_s,
                size_scale=s_scale,
                ndim = ndim,
                channel_dim = channel_dim,
            )
            modules.append(block)

            # update in_channels and in_size for next block
            in_channels = b._last_config.out_channels
            in_size = b._last_config.out_size

        assert out_channels == self.blocks[-1]._last_config.out_channels, (out_channels, self.blocks[-1]._last_config)
        assert out_size == self.blocks[-1]._last_config.out_size,  (out_size, self.blocks[-1]._last_config)
        return torch.nn.Sequential(*modules)


class Lambda(Block):
    def __init__(self, func):
        super().__init__()
        self.func = maybe_compose(func)

    def _make(self, in_channels, out_channels, channel_scale, in_size, out_size, size_scale, ndim, channel_dim):
        if isinstance(self.func, type): return ensure_module(self.func())
        return ensure_module(self.func)

def ensure_block(b: BlockType) -> Block:
    if isinstance(b, Block): return b

    if isinstance(b, (Sequence, Iterable)):
        return Sequential([ensure_block(i) for i in b], -1, -1)

    return Lambda(b)