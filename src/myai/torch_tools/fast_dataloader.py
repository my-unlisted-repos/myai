from collections import abc
import itertools
import math

import numpy as np
import torch

from ..python_tools.types_ import SupportsLenAndGetitem
from ..rng import RNG


class FastDataLoader:
    def __init__(self, data: SupportsLenAndGetitem, batch_size: int, shuffle=False, seed: int | RNG | None = None):
        """Collating dataloader that is usually faster than pytorch, due to having no functionality that I never use.

        Like in pytorch, `data` can define `__getitems__` which accepts a list of indexes and may load them via multiprocessing.

        Args:
            data (SupportsLenAndGetitem): dataset that must support len and indexing.
            batch_size (int): how many samples per batch to load.
            shuffle (bool, optional): set to True to have the data reshuffled at every epoch. Defaults to False.
            seed (bool, optional): seed for random shuffling. None for random seed. Defaults to None.

        Performance:
            At 512 batch size this is 10% faster, at 32 - 2 times faster, at 2 - 4 times faster.
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._use_getitems = hasattr(self.data, "__getitems__")

        if isinstance(seed, RNG): self.rng = seed
        else: self.rng = RNG(seed)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __iter__(self):

        if self.shuffle: indices = self.rng.numpy.permutation(len(self.data))
        else: indices = range(len(self.data))

        for batch_indices in itertools.batched(indices, self.batch_size):
            if self._use_getitems: uncollated_batch = self.data.__getitems__(batch_indices) # type:ignore
            else: uncollated_batch = [self.data[i] for i in batch_indices]
            if isinstance(uncollated_batch[0], torch.Tensor):
                yield torch.stack(uncollated_batch)
            else:
                collated = list(zip(*uncollated_batch))
                yield [torch.stack(i) if isinstance(i[0], torch.Tensor) else torch.tensor(i) for i in collated]



class InMemoryDataloader:
    def __init__(
        self,
        data: torch.Tensor | abc.Sequence[torch.Tensor],
        batch_size: int,
        shuffle=False,
        seed: int | RNG | None = None,
        memory_efficient=False,
    ):
        """Even faster in-memory dataset. There are two versions,
        and the version depends on on `memory_efficient`.

        Args:
            data (torch.Tensor | abc.Sequence[torch.Tensor]):
                either a tensor or a sequence of tensors (like inputs, targets).
            batch_size (int): how many samples per batch to load.
            shuffle (bool, optional): set to True to have the data reshuffled at every epoch. Defaults to False.
            seed (bool, optional): seed for random shuffling. None for random seed. Defaults to None.
            memory_efficient (bool, optional):
                if True, doesn't use any additional memory but is slightly slower.
                If False, uses two times the memory that `data` takes, but is faster.
        """

        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.memory_efficient = memory_efficient

        self._istensor = isinstance(self.data, torch.Tensor)
        self.device = self.data.device if isinstance(self.data, torch.Tensor) else self.data[0].device

        if isinstance(seed, RNG): self.rng = seed
        else: self.rng = RNG(seed)

    def data_length(self):
        ref = self.data if self._istensor else self.data[0]
        return ref.size(0) # type:ignore

    def __len__(self):
        return math.ceil(self.data_length() / self.batch_size)

    def _fast_iter(self):
        if self.shuffle:
            idxs = torch.randperm(self.data_length(), generator = self.rng.torch('cpu')).to(self.device)
            if self._istensor:
                self.data = torch.index_select(self.data, 0, idxs) # type:ignore
            else:
                self.data = [torch.index_select(i, 0, idxs) for i in self.data] # type:ignore

        if self._istensor:
            yield from self.data.split(self.batch_size) # type:ignore
        else:
            yield from zip(*(i.split(self.batch_size) for i in self.data))

    def _memory_efficient_iter(self):
        if self.shuffle:
            idxs = torch.randperm(self.data_length(), generator = self.rng.torch('cpu')).to(self.device)

            for batch_indices in idxs.split(self.batch_size):
                if self._istensor:
                    yield self.data[batch_indices] # type:ignore
                else:
                    yield [i[batch_indices] for i in self.data]

    def __iter__(self):
        if self.memory_efficient: return self._memory_efficient_iter()
        return self._fast_iter()
