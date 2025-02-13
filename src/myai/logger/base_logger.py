import logging
import typing as T
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, MutableMapping

import numpy as np
import torch

from ..plt_tools._types import _K_Collection, _K_Line2D
from ..plt_tools.fig import Fig



def numel(x:np.ndarray | torch.Tensor):
    if isinstance(x, np.ndarray): return x.size
    return x.numel()
class BaseLogger(MutableMapping[str, T.Any], ABC):

    @abstractmethod
    def log(self, step: int, metric: str, value: T.Any) -> None:
        """Log metric value for a given step."""
        # for example
        # self[metric][step] = value

    #mutable mapping requires
    # __getitem__, __setitem__, __delitem__, __iter__, __len__

    @abstractmethod
    def __getitem__(self, metric: str) -> dict[int, T.Any]:
        """Get a `{step: value}` dict for a given metric."""

    @abstractmethod
    def __setitem__(self, metric: str, value: dict[int, T.Any]) -> None:
        """Set a `{step: value}` dict for a given metric."""

    @abstractmethod
    def __delitem__(self, metric: str) -> None:
        """Delete a metric."""

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """Iterate over all metrics."""

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of metrics."""

    @abstractmethod
    def copy(self): return self

    def first(self, key:str):
        return self.get_metric_as_list(key)[0]

    def last(self, key:str):
        return self.get_metric_as_list(key)[-1]

    def min(self, key:str):
        return np.nanmin(self.get_metric_as_numpy(key))

    def max(self, key:str):
        return np.nanmax(self.get_metric_as_numpy(key))

    def mean(self, key: str):
        return np.nanmean(self.get_metric_as_numpy(key))

    def argmin(self, key:str):
        idx = np.nanargmin(self.get_metric_as_numpy(key)).item()
        return self.get_metric_steps(key)[idx]

    def argmax(self, key:str):
        idx = np.nanargmax(self.get_metric_as_numpy(key)).item()
        return self.get_metric_steps(key)[idx]

    def get_metric_as_list(self, key: str) -> list[T.Any]:
        """Returns items under `key` as list. Step is ignored."""
        return list(self[key].values())

    def get_metric_as_numpy(self, key: str) -> np.ndarray:
        """Returns items under `key` as numpy array. Step is ignored."""
        return np.array(self.get_metric_as_list(key))

    def get_metric_as_tensor(self, key: str, dtype = None, device = None) -> torch.Tensor:
        """Returns items under `key` as tensor. Step is ignored."""
        return torch.tensor(self.get_metric_as_list(key), dtype=dtype, device=device)

    def get_metric_steps(self, key: str) -> list[int]:
        """Returns steps for a given key."""
        return list(self[key].keys())

    def get_metric_fill_missing(self, key: str, fill: T.Any = np.nan) -> list[T.Any]:
        """Returns a list of values for a given key, filling missing steps with `fill`, so index of each element is it's step."""
        steps = range(self.num_steps())
        existing = self[key]
        return [(existing[step] if step in existing else fill) for step in steps]

    def get_metric_interpolate(self, key: str) -> np.ndarray:
        """Returns a list of values for a given key, interpolating missing steps."""
        steps = range(self.num_steps())
        existing = self[key]
        return np.interp(steps, list(existing.keys()), list(existing.values()))

    def get_shared_metrics(self, *keys:str | None):
        """Finds all steps where all `keys` exist, returns lists of values for each key. Non string keys are returned as is."""
        arr = np.array([self.get_metric_fill_missing(key, np.nan) for key in keys if isinstance(key,str)]).T
        mask: np.ndarray = np.ma.fix_invalid(arr).mask
        # if there are no nan values, mask will be a 0 ndim array
        if mask.ndim > 1: valid_arr = arr[~mask.any(axis=1)].T
        else: valid_arr = arr.T

        values = []
        i = 0
        for key in keys:
            if isinstance(key, str):
                values.append(valid_arr[i])
                i += 1
            else: values.append(key)
        return values

    def num_steps(self):
        """Returns the largest step recorded."""
        return max(len(i) for i in self.values())

    def state_dict(self):
        return dict(self)

    def load_state_dict(self, sd: Mapping):
        for k,v in sd.items():
            self[k] = v

    # those methods can be overridden if needed
    def save(self, filepath: str):
        """Save this logger to a compressed numpy array file (npz)."""
        arrays = {}
        for k in self.keys():
            try:
                arrays[f"__STEPS__ {k}"] = np.asarray(self.get_metric_steps(k))
                arrays[f"__VALUES__ {k}"] = self.get_metric_as_numpy(k)
            except Exception as e: # pylint:disable=W0718
                logging.warning("Failed to save `%s`: %s", k, e)
        np.savez_compressed(filepath, **arrays)

    def load(self, filepath: str):
        """Load data from a compressed numpy array file (npz) to this logger."""
        arrays = np.load(filepath)
        for k, array in arrays.items():
            if k.startswith('__STEPS__ '):
                name = k.replace("__STEPS__ ", "")
                values = arrays[f"__VALUES__ {name}"]
                self[name] = dict(zip(array, values))

    @classmethod
    def from_file(cls, filepath: str):
        """Load data from a compressed numpy array file (npz) to this logger."""
        logger = cls()
        logger.load(filepath)
        return logger

    def plot(self, *metrics: str, fig = None, **kwargs: T.Unpack[_K_Line2D]):
        k: dict[str, T.Any] = kwargs.copy() # type:ignore # this is necesary for pylance to shut up
        if fig is None: fig = Fig()
        #fig.add()
        ylabel = metrics[0] if len(metrics) == 1 else "value"
        for metric in metrics:
            x = list(self[metric].keys())
            y = list(self[metric].values())
            fig.linechart(x, y, label = metric, **k).axlabels('step', ylabel)
        if len(metrics) > 1: fig.legend()
        return fig.ticks().grid()

    def linechart(self, x:str, y:str, fig = None, **kwargs: T.Unpack[_K_Line2D]):
        xvals = self.get_metric_interpolate(x)
        yvals = self.get_metric_interpolate(y)
        if fig is None: fig = Fig()
        return fig.linechart(xvals, yvals, **kwargs).axlabels(x, y).ticks().grid()

    def as_yaml_string(self):
        # TODO potentially make this better or not
        text = ""
        for key in sorted(self.keys()):
            last = self.last(key)
            text += f"{key}:\n"
            text += f"    count: {len(self[key])}\n"
            text += f"    type: {type(last)}\n"
            if isinstance(last, (torch.Tensor, np.ndarray)) and numel(last) > 1:
                text += f"    last dtype: {last.dtype}\n"
                text += f"    last ndim: {last.ndim}\n"
                text += f"    last shape: {last.shape}\n"
                text += f"    last min: {last.min()}\n"
                text += f"    last max: {last.max()}\n"
                text += f"    last mean: {last.mean()}\n"
                text += f"    last var: {last.var()}\n"
                text += f"    last std: {last.std()}\n"
                text += f"    elements: {numel(last)}\n"
            elif isinstance(last, (int, float, np.ScalarType)) or (isinstance(last, (torch.Tensor, np.ndarray)) and numel(last) == 1):
                values = self.get_metric_as_numpy(key)
                text += f"    last value: {float(last)}\n" # type:ignore
                text += f"    lowest: {values.min()}\n"
                text += f"    highest: {values.max()}\n"
                text += f"    mean: {values.mean()}\n"
            text += "\n"

        return text
