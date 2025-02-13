import itertools
import time
import typing as T
import warnings
from collections import abc

import torch

from ...event_model import Callback, CancelContext

if T.TYPE_CHECKING:
    from ..learner import Learner


class StopOnStep(Callback):
    order = 10000

    def __init__(self,step: int):
        self.step = step

    def after_train_step(self, learner: "Learner"):
        if learner.num_forwards >= self.step:
            raise CancelContext('fit')