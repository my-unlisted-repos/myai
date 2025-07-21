import concurrent.futures
import operator
import warnings
from collections import UserList
from collections.abc import Callable, Sequence, Iterable, Mapping, MutableMapping, MutableSequence
from typing import Any, TypeVar, overload

import numpy as np
import torch
from light_dataloader import LightDataLoader, TensorDataLoader
from torch.utils.data import DataLoader

from ..python_tools import (
    BoundItemGetter,
    Composable,
    SupportsIter,
    SupportsLenAndGetitem,
    compose,
    func2method,
    maybe_compose,
)
from ..utils import describe
from ..rng import RNG, Seed

def _numpy_to(array: np.ndarray, dtype):
    """if dtype is None, this is a noop like in pytorch"""
    if dtype is None: return array
    return array.astype(dtype)

class Sample:
    def __init__(self, data, loader: Composable | None, transform: Composable | None, call=False):
        self._data = data
        self.loader: Callable = maybe_compose(loader)
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

    def append_loader_(self, loader: Composable):
        self.loader = compose(self.loader, loader)
    def prepend_loader_(self, loader: Composable):
        self.loader = compose(loader, self.loader)
    def set_loader_(self, loader: Composable | None):
        self.loader = maybe_compose(loader)

    def append_transform_(self, transform: Composable):
        self.transform = compose(self.transform, transform)
    def prepend_transform_(self, transform: Composable):
        self.transform = compose(transform, self.transform)
    def set_transform_(self, transform: Composable | None):
        self.transform = maybe_compose(transform)

    def copy(self):
        sample = self.__class__(data=self._data, loader=self.loader, transform=self.transform, call=self.call)
        sample.preloaded = self.preloaded
        return sample

# picks a random sample from choices
class _RandomChoice:
    __slots__ = ("samples", "rng")
    def __init__(self, samples: Sequence[Sample], seed = None):
        self.samples = samples
        self.rng = RNG(seed)
    def __call__(self): return self.rng.random.choice(self.samples)()

# class DS[R](abc.Sequence[R]):
R = TypeVar("R")
X = TypeVar("X")

class DS(torch.utils.data.Dataset[R]):
    def __init__(self, n_threads = 0):
        super().__init__()
        self.samples: list[Sample] = []

        self.n_threads = n_threads
        if n_threads > 0: self._executor = concurrent.futures.ThreadPoolExecutor(n_threads)
        else: self._executor = None

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __getitem__(self, index: int) -> R:
        return self.samples[index]()

    def __getitems__(self, indexes: Iterable[int]) -> list[R]:
        if self._executor is not None:
            return list(self._executor.map(lambda i: self.samples[i](), indexes))

        return [self.samples[i]() for i in indexes]

    def __iter__(self):
        for sample in self.samples: yield sample()

    def __len__(self):
        return len(self.samples)

    def _add_sample_object_(self, sample: Sample):
        self.samples.append(sample)
        return self

    def _add_sample_objects_(self, samples: Iterable[Sample]):
        self.samples.extend(samples)
        return self

    def add_sample_(self, data, loader: Composable | None = None, transform: Composable | None = None):
        self._add_sample_object_(Sample(data, loader, transform))
        return self

    def add_samples_(self, samples: SupportsIter, loader: Composable | None = None, transform: Composable | None = None, call=False):
        self._add_sample_objects_(Sample(s, loader, transform, call=call) for s in samples)
        return self

    def add_dataset_(self, dataset: SupportsLenAndGetitem, loader: Composable | None = None, transform: Composable | None = None):
        self._add_sample_objects_(Sample(BoundItemGetter(dataset, i), loader, transform, call=True) for i in range(len(dataset))) # type:ignore
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

    def dataloader(self, batch_size: int, shuffle: bool, **kwargs) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self, batch_size = batch_size, shuffle = shuffle, **kwargs)

    # def dataloader[D:Callable](self, batch_size: int, shuffle: bool, seed: int | None = None, cls: D = LightDataLoader) -> D:
    def light_dataloader(self, batch_size: int, shuffle: bool, seed: int | None = None) -> LightDataLoader:
        return LightDataLoader(self, batch_size = batch_size, shuffle = shuffle, seed=seed)

    def tensor_dataloader(
        self,
        batch_size: int,
        shuffle: bool,
        memory_efficient: bool = False,
        seed: int | None = None,
        dtype=None,
        device=None,
    ):
        return TensorDataLoader(
            self.stack_tensors(dtype=dtype, device=device),
            batch_size=batch_size,
            shuffle=shuffle,
            memory_efficient=memory_efficient,
            seed=seed,
        )

    def stack_tensors(self, dtype=None, device=None) -> Any:
        """returns either a tensor if each sample is a tensor, or a list of tensors if samples are sequences."""
        samples = list(self)
        first = samples[0]

        if isinstance(first, torch.Tensor): return torch.stack(samples).to(dtype=dtype, device=device)
        if isinstance(first, np.ndarray): return torch.as_tensor(np.stack(samples), dtype=dtype, device=device)

        if isinstance(first, Sequence):
            tensors = type(first)() if isinstance(first, MutableSequence) else []
            keys = list(range(len(first)))

        elif isinstance(first, Mapping):
            tensors = type(first)() if isinstance(first, MutableMapping) else {}
            keys = list(first.keys())

        else:
            raise ValueError(f"Samples must be tensor, array, sequence or mapping, got {type(first)}")

        if not isinstance(device, (Sequence, Mapping)): device = {k: device for k in keys}
        if not isinstance(dtype, (Sequence, Mapping)): dtype = {k: dtype for k in keys}

        for k in keys:
            de = device[k]
            dt = dtype[k]

            if isinstance(first[k], torch.Tensor):
                tensor = torch.stack([sample[k] for sample in samples]).to(dtype=dt, device=de)
            else:
                tensor = torch.stack([torch.as_tensor(sample[k], dtype=dt, device=de) for sample in samples])

            if isinstance(tensors, Sequence): tensors.append(tensor)
            else: tensors[k] = tensor

        return tensors

    def stack_numpy(self, dtype=None) -> Any:
        """returns either a numpy array or a list of arrays if samples are sequences."""
        samples = list(self)
        first = samples[0]

        if isinstance(first, torch.Tensor): return _numpy_to(np.stack([s.numpy(force=True) for s in samples]), dtype)
        if isinstance(first, np.ndarray): return _numpy_to(np.stack(samples), dtype)

        if isinstance(first, Sequence):
            arrays = type(first)() if isinstance(first, MutableSequence) else []
            keys = list(range(len(first)))

        elif isinstance(first, Mapping):
            arrays = type(first)() if isinstance(first, MutableMapping) else {}
            keys = list(first.keys())

        else:
            raise ValueError(f"Samples must be tensor, array, sequence or mapping, got {type(first)}")

        if not isinstance(dtype, (Sequence, Mapping)): dtype = {k: dtype for k in keys}

        for k in keys:
            dt = dtype[k]

            if isinstance(first[k], torch.Tensor):
                array = np.stack([_numpy_to(sample[k].numpy(force=True), dt) for sample in samples])
            else:
                array = np.stack([_numpy_to(sample[k], dt) for sample in samples])

            if isinstance(arrays, Sequence): arrays.append(array)
            else: arrays[k] = array

        return arrays

    def append_loader_(self, loader: Composable):
        for s in self.samples: s.append_loader_(loader)
        return self

    def prepend_loader_(self, loader: Composable):
        for s in self.samples: s.prepend_loader_(loader)
        return self

    def set_loader_(self, loader: Composable | None):
        for s in self.samples: s.set_loader_(loader)
        return self

    def append_transform_(self, transform: Composable):
        for s in self.samples: s.append_transform_(transform)
        return self

    def prepend_transform_(self, transform: Composable):
        for s in self.samples: s.prepend_transform_(transform)
        return self

    def set_transform_(self, transform: Composable | None):
        for s in self.samples: s.set_transform_(transform)
        return self

    def shuffle_(self, seed: int | RNG | None = None):
        RNG(seed).random.shuffle(self.samples)
        return self

    def shuffled(self, seed: int | RNG | None = 0, copy_samples = False):
        return self.copy(copy_samples).shuffle_(seed)

    def _to_absolute_amount(self, amount: int | float | None, ):
        """if number of samples is float, it is treated as relative to total number of samples or number. None means all samples."""
        if amount is None: return len(self)
        if isinstance(amount, float): return int(amount * len(self))
        return amount

    def preload_(self, amount: int | float | None = None, clear_data = False):
        """Preloads all or first `amount` samples (uses actual samples, ignores idxs)."""
        amount = self._to_absolute_amount(amount)

        if self._executor is not None:
            list(self._executor.map(operator.methodcaller('preload_'), self.samples[:amount]))

        else:
            for s in self.samples[:amount]: s.preload_()

        if clear_data:
            for s in self.samples[:amount]: s._data = None

        return self

    def split(self, splits: int | float | Sequence[int | float], shuffle = True, seed: Seed = 0) -> "list[DS[R]]":
        if isinstance(splits, (int, float)): splits = [splits, ]

        splits = [self._to_absolute_amount(s) for s in splits]
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

    def _get_samples_per_class(self, target_idx: int | Callable | Any = -1, preload=False) -> dict[Any, list[Sample]]:
        """returns a dictionary that maps each target to a list of indexes of samples with that target"""
        samples_per_class: dict[Any, list[Sample]] = {}

        for sample in self.samples:
            if preload: sample.preload_()
            data = sample()

            if callable(target_idx): target = target_idx(data)
            else: target = data[target_idx]

             # tensors and arrays are unique objects even if they have the same value
            if isinstance(target, torch.Tensor):
                if target.numel() == 1: target = target.detach().cpu().item()
                else: target = tuple(target.detach().cpu().tolist())

            elif isinstance(target, np.ndarray):
                if target.size == 1: target = target.item()
                else: target = tuple(target.tolist())

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
        num_samples = s._to_absolute_amount(num_samples)

        samples_per_class = s._get_samples_per_class(target_idx=target_idx, preload=preload)

        if per_class:
            samples_per_class = {cls: samples[:num_samples] for cls,samples in samples_per_class.items()}
        else:
            num_classes = len(samples_per_class)
            num_samples_per_class = num_samples // num_classes
            samples_per_class = {cls: samples[:num_samples_per_class] for cls,samples in samples_per_class.items()}

        ds = DS()
        for cls in samples_per_class.values():
            ds._add_sample_objects_(cls)

        return ds

    def oversample(self, min_samples: int | None = None, target_idx: int | Callable = -1, preload=False, seed: Seed = 0) -> "DS[R]":
        """makes sure samples for each class are equally likely to appear via padding each class with
        samples that randomly choose a sample of that class.

        Args:
            min_samples (int | None, optional): pad to this number of samples. If None, uses number of samples of the largest class.
            target_idx (int | Callable, optional): index of target or function that returns the target. Defaults to -1.
            preload (bool, optional):
                whether to preload all samples. Preloading them manually may use more memory because
                the duplicates will be pre-loaded separately. Defaults to False.
            seed (Seed, optional): seed. Defaults to None.

        Returns:
            DS[R]: oversampled dataset.
        """
        samples_per_class = self._get_samples_per_class(target_idx=target_idx, preload=preload)
        if min_samples is None: min_samples = max(len(i) for i in samples_per_class.values())

        ds = DS()
        rng = RNG(seed)
        for samples in samples_per_class.values():
            ds._add_sample_objects_(samples)
            remaining = min_samples - len(samples)
            if remaining > 0:
                sampler = _RandomChoice(samples, seed=rng)
                ds.add_samples_([sampler for _ in range(remaining)], call=True)
        return ds

    def label_stats(self, target_idx: int | Callable = -1, preload=False, ) -> dict[Any, int]:
        """Returns a dictionary with number of samples per each class."""
        idxs_per_class = self._get_samples_per_class(target_idx=target_idx, preload=preload)
        return {k: len(v) for k, v in idxs_per_class.items()}

    def calculate_mean_std(
        self,
        dim: int | Sequence[int] | None = 0,
        batch_size = 32,
    ):
        """calculates mean and std of each sample along dim, if samples are sequences, first element is used."""
        if dim is None: dim = []
        elif isinstance(dim, int): dim = [dim]
        else: dim = list(dim)

        dl = torch.utils.data.DataLoader(self, batch_size=batch_size,) # type:ignore

        mean = std = None
        nsamples = 0
        for sample in dl:
            # get the actual sample if there is a label
            if isinstance(sample, (list, tuple)):  sample = sample[0]
            sample: torch.Tensor

            # find the reduction dims
            # d + 1 because we have additional batch dim
            keep_dims = [d+1 for d in dim]
            reduce_dims = [d for d in range(sample.ndim) if d not in keep_dims]

            # create mean and std counters
            if mean is None or std is None:
                if len(dim) == 0:
                    mean = 0; std = 0
                else:
                    mean = torch.zeros([sample.shape[d] for d in keep_dims], device=sample.device, dtype=sample.dtype)
                    std = torch.zeros([sample.shape[d] for d in keep_dims], device=sample.device, dtype=sample.dtype)


            mean += sample.mean(dim=reduce_dims)
            std += sample.std(dim=reduce_dims)
            nsamples += 1

        if mean is None or std is None: raise ValueError('Dataset is empty.')
        return mean / nsamples, std / nsamples

    def describe(self):
        sample = self[0]
        return describe(sample)