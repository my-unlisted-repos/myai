from collections.abc import Sequence

import torch

from .. import B
from ..containers import ModuleList

_Conv3ReLU = B.ConvBlock(3, act = B.Act('relu'))

class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ndim: int = 2,
        channels = (64, 128, 256),
        first: B.BlockType | None = B.Sequential([_Conv3ReLU, _Conv3ReLU], 0, 0),
        first_out_channels = 32,
        down: B.BlockType = _Conv3ReLU,
        middle: B.BlockType | None = _Conv3ReLU,
        up: B.BlockType = B.ConvTransposeBlock(3, act = B.Act('relu')),
        head: B.BlockType = B.ConvBlock(1),
        skip_block: B.BlockType | None = None,
        skip_mode: B.AggregateModes = 'cat',
        # in_size: int | Sequence[int] | None = None,
    ):
        super().__init__()

        cur_channels = in_channels

        # FIRST
        self.first = None
        if first is not None:
            self.first = B.ensure_block(first).new(cur_channels, first_out_channels, ndim=ndim)
            cur_channels = first_out_channels

        # ENCODER
        encoder_modules = []
        skip_modules = []
        for c in channels:
            encoder_modules.append(B.ensure_block(down).new(cur_channels, c, size_scale=0.5, ndim = ndim))
            cur_channels = c
            if skip_block is not None: skip_modules.append(B.ensure_block(skip_block).new(c, c))
        self.encoder = ModuleList(encoder_modules)
        self.skip_modules = ModuleList(skip_modules) if skip_block is not None else None

        # MIDDLE
        self.middle = B.ensure_block(middle).new(cur_channels, cur_channels, ndim = ndim) if middle is not None else None

        # DECODER
        decoder_modules = [B.ensure_block(up).new(cur_channels, channels[-2])]
        cur_channels = channels[-2]
        for c in list(reversed(channels))[1:]:
            decoder_modules.append(B.Aggregate(skip_mode, in_blocks=skip_block, out_block=up).new([cur_channels,cur_channels], c, size_scale=2))
            cur_channels = c
        self.decoder = ModuleList(decoder_modules)

        # HEAD
        self.head = B.ensure_block(head).new(cur_channels, out_channels)

    def forward(self, input:torch.Tensor):
        # FIRST
        if self.first is not None: x = self.first(input)
        else: x = input

        # ENCODER
        skips = [x]
        for i,m in enumerate(self.encoder):
            x = m(x)
            if i != len(self.encoder) - 1: skips.insert(0, x)

        # MIDDLE
        if self.middle is not None: x = self.middle(x)

        # DECODER
        x = self.decoder[0](x)
        for m, s in zip(self.decoder[1:], skips):
            x = m(x, s)

        # HEAD
        return self.head(x)


# SegResBlock = ResidualCube.partial(
#     ChainCube.partial(
#         cube = ConvCube.partial(
#             kernel_size = 3,
#             act = "relu",
#             norm = "bn",
#             order="NADC",
#             bias=False
#         ),
#         num = 2,
#         channel_mode = 'max',
#     )
# )