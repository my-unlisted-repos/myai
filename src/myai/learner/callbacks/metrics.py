import time
import typing as T
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import torch

from ...event_model import Callback
from ...torch_tools.conversion import maybe_detach_cpu
from ...metrics import accuracy, dice, iou, binary_accuracy
from ...torch_tools import batched_raw_preds_to_one_hot

if T.TYPE_CHECKING:
    from ..learner import Learner


class Loss(Callback):
    order = -1
    def __init__(self, agg_fn = np.nanmean, name = 'loss'):
        self.test_losses = []
        self.agg_fn = agg_fn
        self.name = name

    def after_train_step(self, learner: "Learner"):
        learner.log(f"train {self.name}", learner.loss.detach().cpu())

    def after_test_step(self, learner: "Learner"):
        self.test_losses.append(learner.loss.detach().cpu())

    def after_test_epoch(self, learner: "Learner"):
        if len(self.test_losses) > 0:
            learner.log(f"test {self.name}", self.agg_fn(self.test_losses))
            self.test_losses = []


class LogTime(Callback):
    order = 1000
    def __init__(self):
        self.start_time = None
    def after_train_step(self, learner: "Learner"):
        t = time.time()
        if self.start_time is None: self.start_time = t
        learner.log("time", t-self.start_time)


class Metric(Callback, ABC):
    order = -1
    """A metric that gets logged after train and test batches.
    Please make sure order is not bigger than 0, reason being that
    some callbacks might run stuff through the model and assign `preds` and `targets`.
    So the metric won't run with the actual train or test samples."""
    def __init__(self, metric: str, train_step: int | None, test_step: int | None, agg_fn = np.mean):
        super().__init__()
        self.metric = metric
        self.train_step = train_step
        self.test_step = test_step
        self.agg_fn = agg_fn

        self.test_epoch_values = []
        if self.order > 0:
            warnings.warn(
                f"Metric {self.metric} has order {self.order} which is higher than 0, so it might run after callbacks that modify `learner.preds`, `learner.targets`, etc."
            )

    @abstractmethod
    def __call__(self, learner: "Learner") -> T.Any:
        """Evaluate the metric. Please make sure returned value is detached and on CPU."""

    def after_train_step(self, learner: "Learner"):
        if self.train_step is not None and learner.num_forwards % self.train_step == 0:
            learner.log(f'train {self.metric}', self(learner))

    def after_test_step(self, learner: "Learner"):
        if self.test_step is not None and learner.cur_batch % self.test_step == 0:
            self.test_epoch_values.append(self(learner))

    def after_test_epoch(self, learner: "Learner"):
        if len(self.test_epoch_values) > 0:
            learner.log(f'test {self.metric}', self.agg_fn(self.test_epoch_values))
            self.test_epoch_values = []

class Accuracy(Metric):
    def __init__(self, train_step: int | None = 1, test_step: int | None = 1, agg_fn = np.mean, name = 'accuracy'):
        super().__init__(name, train_step, test_step, agg_fn)

    def __call__(self, learner: "Learner") -> float:
        return accuracy(batched_raw_preds_to_one_hot(learner.preds), learner.targets).detach().cpu().item()

class BinaryAccuracy(Metric):
    def __init__(self, threshold, train_step: int | None = 1, test_step: int | None = 1, agg_fn = np.mean, name = 'accuracy'):
        super().__init__(name, train_step, test_step, agg_fn)
        self.threshold = threshold

    def __call__(self, learner: "Learner") -> float:
        return binary_accuracy(learner.preds > self.threshold, learner.targets.bool()).detach().cpu().item()

class PerClassMetric(Callback, ABC):
    order = -1
    """A metric that gets logged after train and test batches.
    Please make sure order is not bigger than 0, reason being that
    some callbacks might run stuff through the model and assign `preds` and `targets`.
    So the metric won't run with the actual train or test samples."""

    def __init__(
        self,
        metric: str,
        class_labels,
        train_step: int | None,
        test_step: int | None,
        bg_index: int | None,
    ):
        """_summary_

        :param metric: _description_
        :param class_labels: This INCLUDES BACKGROUND!
        :param train_step: _description_
        :param test_step: _description_
        :param bg_index: _description_, defaults to 0
        :param agg_fn: _description_, defaults to np.mean
        """
        super().__init__()
        self.metric = metric
        self.train_step = train_step
        self.test_step = test_step
        self.class_labels = class_labels
        self.bg_index = bg_index

        self.test_epoch_values = []
        if self.order > 0:
            warnings.warn(
                f"Metric {self.metric} has order {self.order} which is higher than 0, so it might run after callbacks that modify `learner.preds`, `learner.targets`, etc."
            )

    @abstractmethod
    def __call__(self, learner: "Learner") -> T.Any:
        """Evaluate the metric. Please make sure returned value is detached and on CPU."""

    def after_train_step(self, learner: "Learner"):
        if self.train_step is not None and learner.num_forwards % self.train_step == 0:
            values = self(learner)

            if self.class_labels is None: self.class_labels = list(range(len(values)))

            for label, value in zip(self.class_labels, values):
                learner.log(f'train {self.metric} - {label}', value)

            if self.bg_index is not None: values = np.delete(values, self.bg_index)

            learner.log(f'train {self.metric} mean', np.nanmean(values))

    def after_test_step(self, learner: "Learner"):
        if self.test_step is not None and learner.cur_batch % self.test_step == 0:
            self.test_epoch_values.append(self(learner))

    def after_test_epoch(self, learner: "Learner"):
        if len(self.test_epoch_values) > 0:

            values = np.nanmean(self.test_epoch_values, 0)

            if self.class_labels is None: self.class_labels = list(range(len(values)))

            for label, value in zip(self.class_labels, values):
                learner.log(f'test {self.metric} - {label}', value)

            if self.bg_index is not None: values = np.delete(values, self.bg_index)
            learner.log(f'test {self.metric} mean', np.nanmean(values))

            self.test_epoch_values = []


class Dice(PerClassMetric):
    def __init__(
        self,
        class_labels=None,
        bg_index=None,
        train_step: int | None = 1,
        test_step: int | None = 1,
        name="dice",
        binary_threshold = 0.,
    ):
        """Dice metric. For binary and multiclass.

        Args:
            class_labels (_type_, optional): _description_. Defaults to None.
            bg_index (_type_, optional): _description_. Defaults to None.
            train_step (int | None, optional): _description_. Defaults to 1.
            test_step (int | None, optional): _description_. Defaults to 1.
            name (str, optional): _description_. Defaults to "dice".
        """
        super().__init__(
            metric=name,
            class_labels=class_labels,
            train_step=train_step,
            test_step=test_step,
            bg_index=bg_index,
        )
        self.binary_threshold = binary_threshold

    def __call__(self, learner: "Learner"):
        n_channels = learner.preds.shape[1]
        if n_channels > 1: preds = batched_raw_preds_to_one_hot(learner.preds)
        else: preds = learner.preds > self.binary_threshold
        return dice(preds, learner.targets).detach().cpu()

class IOU(PerClassMetric):
    def __init__(
        self,
        class_labels=None,
        bg_index=None,
        train_step: int | None = 1,
        test_step: int | None = 1,
        name="iou",
        binary_threshold = 0.,
    ):
        super().__init__(
            metric=name,
            class_labels=class_labels,
            train_step=train_step,
            test_step=test_step,
            bg_index=bg_index,
        )
        self.binary_threshold = binary_threshold

    def __call__(self, learner: "Learner"):
        n_channels = learner.preds.shape[1]
        if n_channels > 1: preds = batched_raw_preds_to_one_hot(learner.preds)
        else: preds = learner.preds > self.binary_threshold
        return iou(preds, learner.targets).detach().cpu()