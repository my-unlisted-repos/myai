#pylint:disable=redefined-outer-name
import itertools
import time
import typing as T
import warnings
from collections import abc

import torch

from ...event_model import Callback, CancelContext

if T.TYPE_CHECKING:
    from ..learner import Learner

class _SchedulerCallback(Callback):
    def __init__(self, scheduler,):
        self.scheduler = scheduler

    def enter(self, learner: "Learner"):
        if self.scheduler is not None:
            learner._set_x('scheduler', self.scheduler)
            self.scheduler = learner.scheduler
        else:
            learner.scheduler = None

class BatchScheduler(_SchedulerCallback):
    def after_train_batch(self, learner: "Learner"):
        if self.scheduler is not None: self.scheduler.step()

class EpochScheduler(_SchedulerCallback):
    def after_train_epoch(self, learner: "Learner"):
        if self.scheduler is not None: self.scheduler.step()

class StepScheduler(_SchedulerCallback):
    def after_train_step(self, learner: "Learner"):
        if self.scheduler is not None: self.scheduler.step()

def scheduler(scheduler, on: T.Literal['batch', 'epoch', 'step'] = 'batch',):
    if on == 'batch':
        return BatchScheduler(scheduler)
    if on == 'epoch':
        return EpochScheduler(scheduler)
    if on == 'step':
        return StepScheduler(scheduler)
    raise ValueError(on)