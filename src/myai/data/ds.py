    # pylint: disable=undefined-variable
# because its bugged with generics
import concurrent.futures
import operator
from collections import UserList, abc
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
import torch
from light_dataloader import LightDataLoader, TensorDataLoader
from torch.utils.data import DataLoader

from ..python_tools import (
    Composable,
    SupportsIter,
    SupportsLenAndGetitem,
    compose,
    func2method,
    maybe_compose,
)
from ..rng import RNG, Seed


class Sample:
    def __init__(self, data, loader: Composable | None, transform: Composable | None, call=False):
        self._data = data
        self.loader: abc.Callable = maybe_compose(loader)
        self.transform = maybe_compose(transform)
        self.call = call

        self.preloaded = None

    @property
    def data(self):
        if self.call: return self._data()
        return self._data

    def __call__(self):
        if self.preloaded is not None: return self.transform(self.preloaded)
        return self.transform(self.loader(self.data))

    def preload_(self):
        if self.preloaded is None: self.preloaded = self.loader(self.data)

    def unload_(self):
        self.preloaded = None

    def add_loader_(self, loader: Composable):
        self.loader = compose(self.loader, loader)
    def set_loader_(self, loader: Composable | None):
        self.loader = maybe_compose(loader)

    def add_transform_(self, transform: Composable):
        self.transform = compose(self.transform, transform)
    def set_transform_(self, transform: Composable | None):
        self.transform = maybe_compose(transform)

    def copy(self):
        sample = self.__class__(self.data, self.loader, self.transform)
        sample.preloaded = self.preloaded
        return sample

class _RandomChoice:
    __slots__ = ("data", "rng")
    def __init__(self, data: Sequence[Sample], seed = None):
        self.data = data
        self.rng = RNG(seed)
    def __call__(self): return self.rng.random.choice(self.data)()

# class DS[R](abc.Sequence[R]):
R = TypeVar("R")
X = TypeVar("X")
class DS(abc.Sequence[R]):
    def __init__(self, n_threads = 0):
        super().__init__()
        self.samples: list[Sample] = []
        self.idxs: list[int] = []

        self.n_threads = n_threads
        if n_threads > 0: self._executor = concurrent.futures.ThreadPoolExecutor(n_threads)
        else: self._executor = None

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __getitem__(self, idx) -> R: # type:ignore
        return self.samples[self.idxs[idx]]()

    def __getitems__(self, indexes: abc.Iterable[int]) -> list[R]:
        if self._executor is not None:
            return list(self._executor.map(lambda i: self.samples[self.idxs[i]](), indexes))

        return [self.samples[self.idxs[i]]() for i in indexes]

    def __iter__(self):
        for idx in self.idxs: yield self.samples[idx]()

    def __len__(self):
        return len(self.samples)

    def _get_sample(self, idx: int):
        return self.samples[self.idxs[idx]]

    def _yield_samples(self):
        for idx in self.idxs: yield self.samples[idx]

    def _add_sample_object_(self, sample: Sample):
        self.idxs.append(len(self.samples))
        self.samples.append(sample)
        return self

    def _add_sample_objects_(self, samples: list[Sample]):
        self.idxs.extend(range(len(self.samples), len(self.samples) + len(samples)))
        self.samples.extend(samples)
        return self

    def add_sample_(self, data, loader: Composable | None = None, transform: Composable | None = None):
        self._add_sample_object_(Sample(data, loader, transform))
        return self

    def add_samples_(self, samples: SupportsIter, loader: Composable | None = None, transform: Composable | None = None, call=False):
        self._add_sample_objects_([Sample(s, loader, transform, call=call) for s in samples])
        return self

    def add_dataset_(self, dataset: SupportsLenAndGetitem, loader: Composable | None = None, transform: Composable | None = None):
        self._add_sample_objects_([Sample(operator.getitem(dataset, i), loader, transform, call=True) for i in range(len(dataset))]) # type:ignore
        return self

    def merge_(self, ds: "DS"):
        self._add_sample_objects_(ds.samples)
        return self

    def merged_with(self, ds: "DS"):
        merged = self.__class__(n_threads = self.n_threads)
        merged._add_sample_objects_(self.samples)
        merged._add_sample_objects_(ds.samples)
        return merged

    def copy(self, copy_samples = True):
        ds = self.__class__(n_threads = self.n_threads)

        if copy_samples: samples = [i.copy() for i in self.samples]
        else: samples = self.samples.copy()

        ds._add_sample_objects_(samples)
        return ds

    # def dataloader[D:Callable](self, batch_size: int, shuffle: bool, seed: int | None = None, cls: D = LightDataLoader) -> D:
    def dataloader(self, batch_size: int, shuffle: bool, seed: int | None = None, cls: type[X] = LightDataLoader) -> X:
        return cls(self, batch_size = batch_size, shuffle = shuffle, seed=seed) # pyright:ignore[reportCallIssue]

    def stack(self, dtype=None, device=None):
        samples = list(i for i in self)
        if isinstance(samples[0], torch.Tensor): tensors = torch.stack(samples).to(dtype=dtype, device=device)
        else:
            if not isinstance(dtype, Sequence): dtype = [dtype for _ in range(len(samples[0]))]
            if not isinstance(device, Sequence): device = [device for _ in range(len(samples[0]))]
            tensors = [torch.stack([sample[i] if isinstance(sample[i], torch.Tensor) else torch.as_tensor(sample[i]) for sample in samples]).to(dtype=dt, device = de) for i, dt, de in zip(range(len(samples[0])), dtype, device)]
        return tensors

    def tensor_dataloader(self, batch_size: int, shuffle: bool, memory_efficient:bool = False, seed: int | None = None, dtype = None, device=None):
        return TensorDataLoader(self.stack(dtype=dtype, device=device), batch_size = batch_size, shuffle = shuffle, memory_efficient=memory_efficient, seed=seed)

    def add_loader_(self, loader: Composable):
        for s in self.samples: s.add_loader_(loader)
        return self

    def set_loader_(self, loader: Composable | None):
        for s in self.samples: s.set_loader_(loader)
        return self

    def add_transform_(self, transform: Composable):
        for s in self.samples: s.add_transform_(transform)
        return self

    def set_transform_(self, transform: Composable | None):
        for s in self.samples: s.set_transform_(transform)
        return self

    def shuffle_(self, seed: int | RNG | None = None):
        RNG(seed).random.shuffle(self.idxs)
        return self

    def shuffled(self, seed: int | RNG | None, copy_samples = False):
        return self.copy(copy_samples).shuffle_(seed)

    def _ensure_absolute_amount(self, amount: int | float | None):
        if amount is None: return len(self)
        if isinstance(amount, float): return int(amount * len(self))
        return amount

    def preload_(self, amount: int | float | None = None, clear_data = False):
        """Preloads all or first `amount` samples."""
        amount = self._ensure_absolute_amount(amount)

        if self._executor is not None:
            with self._executor as ex:
                for _ in ex.map(operator.methodcaller('preload'), self.samples[:amount]): pass
        else:
            for s in self.samples[:amount]: s.preload_()

        if clear_data:
            for s in self.samples[:amount]: s._data = None

        return self

    def split(self, splits: int | float | abc.Sequence[int | float], shuffle = True, seed: Seed = 0) -> "list[DS[R]]":
        if isinstance(splits, (int, float)): splits = [splits, ]

        splits = [self._ensure_absolute_amount(s) for s in splits]
        if len(splits) == 1: splits.append(len(self) - splits[0])

        idxs = list(range(len(self.samples)))
        if shuffle:
            RNG(seed).random.shuffle(idxs)

        datasets = [DS(self.n_threads) for _ in splits]

        cur = 0
        for i, s in enumerate(splits):
            datasets[i]._add_sample_objects_([self.samples[i] for i in idxs[cur:cur+s]])
            cur += s

        return datasets

    def _get_samples_per_class(self, target_idx: int | Callable = -1, preload=False) -> dict[Any, list[Sample]]:
        samples_per_class: dict[Any, list[Sample]] = {}

        for sample in self.samples:
            if preload: sample.preload_()
            data = sample()

            if callable(target_idx): target = target_idx(data)
            else: target = data[target_idx]

             # tensors are unique objects even if they have the same value
            if isinstance(target, torch.Tensor):
                assert target.numel() == 1, target.shape
                target = target.detach().cpu().item()
            elif isinstance(target, np.ndarray):
                assert target.size == 1, target.shape
                target = target.item()

            if target in samples_per_class: samples_per_class[target].append(sample)
            else: samples_per_class[target] = [sample]

        return samples_per_class

    def subsample(
        self,
        num_samples: int | float,
        per_class: bool = False,
        target_idx: int | Callable = -1,
        preload=False,
        shuffle=True,
        seed: Seed = 0,
    ) -> "DS[R]":
        """Subsample this dataset.

        Args:
            samples (int | float): number of samples, absolute int or relative float.
            per_class (bool, optional): whether number of samples is per class or in total. Defaults to False.
            target_idx (int | Callable, optional): index of target or function that returns the target. Defaults to -1.
            preload (bool, optional): preloads all samples. Subsampling loads all samples to get their labels, so might as well
                store the loaded data. Defaults to False.
            shuffle (bool, optional): whether to shuffle before subsampling. Defaults to True.
            seed (int, optional): seed for shuffling. Defaults to 0.

        Returns:
            DS[R]: subsampled dataset.
        """
        s = self.shuffled(seed) if shuffle else self

        samples_per_class: dict[Any, list[Sample]] = self._get_samples_per_class(target_idx=target_idx, preload = preload)

        if per_class:
            if isinstance(num_samples, float): num_samples = int(num_samples * len(s))
            samples_per_class = {k: v[:num_samples] for k,v in samples_per_class.items()}
        else:
            num_classes = len(samples_per_class)
            num_samples_per_class = num_samples // num_classes

            if isinstance(num_samples, float): num_samples = int(num_samples * num_samples_per_class)
            samples_per_class = {k: v[:num_samples_per_class] for k,v in samples_per_class.items()}

        ds = DS()
        for cls in samples_per_class.values():
            ds._add_sample_objects_(cls)
        return ds

    def oversample(self, target_idx: int | Callable = -1, preload=False, seed: Seed = None) -> "DS[R]":
        """makes sure samples for each class are equally likely to appear via padding each class with
        samples that randomly choose a sample of that class."""
        samples_per_class: dict[Any, list[Sample]] = self._get_samples_per_class(target_idx=target_idx, preload = preload)
        max_samples_per_class = max(len(i) for i in samples_per_class.values())

        ds = DS()
        for cls in samples_per_class.values():
            ds._add_sample_objects_(cls)
            remaining = max_samples_per_class - len(cls)
            if remaining > 0:
                sampler = _RandomChoice(cls, seed = seed)
                ds.add_samples_([sampler for _ in range(remaining)], call=True)
        return ds

    def label_stats(self, target_idx: int | Callable = -1, preload=False, ) -> dict[Any, int]:
        """Returns a dictionary with number of samples per each class."""
        samples_per_class: dict[Any, list[Sample]] = self._get_samples_per_class(target_idx=target_idx, preload = preload)
        return {k: len(v) for k, v in samples_per_class.items()}

    def calculate_mean_std(
        self,
        dim: int | list[int] | tuple[int] | None = 0,
        batch_size = 32,
    ):
        if dim is None: dim = []
        elif isinstance(dim, int): dim = [dim]

        dl = torch.utils.data.DataLoader(self, batch_size=batch_size,) # type:ignore

        mean = std = None
        nsamples = 0
        for sample in dl:
            # get the actual sample if there is a label
            if isinstance(sample, (list, tuple)):  sample = sample[0]
            sample: torch.Tensor

            # find the reduction dims
            # d + 1 because we have additional batch dim
            other_dims = [d for d in range(sample.ndim) if d+1 not in dim]

            # create mean and std counters
            if mean is None or std is None:
                if len(dim) == 0:
                    mean = 0; std = 0
                else:
                    mean = torch.zeros([sample.shape[d+1] for d in dim])
                    std = torch.zeros([sample.shape[d+1] for d in dim])


            mean += sample.mean(dim=other_dims)
            std += sample.std(dim=other_dims)
            nsamples += 1

        if mean is None or std is None: raise ValueError('Dataset is empty.')
        return mean / nsamples, std / nsamples
