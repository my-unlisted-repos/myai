import itertools
import time
import typing as T
import warnings
from collections import abc

import torch

from ...event_model import Callback, CancelContext

if T.TYPE_CHECKING:
    from ..learner import Learner


class NoOp(Callback): pass

class NoClosure(Callback):
    """Don't pass closure to optimizer."""
    def enter(self, learner: "Learner"):
        learner.set_use_closure(False)
    def exit(self, learner: "Learner"):
        learner.set_use_closure(True)


class BatchTfms(Callback):
    def __init__(self, tfms):
        self.tfms = tfms

    def before_train_batch(self, learner: "Learner"):
        learner.batch = self.tfms(learner.batch)
