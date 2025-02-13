from collections import abc
import typing
from ..data import DS

def make_even_subsampled_dataset(dataset: abc.Iterable[tuple[typing.Any, typing.Any]], samples_per_class: int):
    """Dataset must be an iterable of (sample, class) where class is anything
    that can be a key in a dict so can be DS.
    This creates a new DS with first samples_per_class samples per each class."""
    samples: dict[typing.Any, list] = {}
    for img, label in dataset:
        if label in samples:
            if len(samples[label]) == samples_per_class: continue
            samples[label].append((img, label))
        else: samples[label] = [(img, label)]

    ds = DS()
    for v in samples.values():
        ds.add_samples_(v)
    return ds