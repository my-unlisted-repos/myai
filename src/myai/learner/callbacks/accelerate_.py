from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator

from ...event_model import Callback

if TYPE_CHECKING:
    from ..learner import Learner
    from .val import TestEpoch

class Accelerate(Callback):
    order = -20
    def __init__(self, mixed_precision = None, cpu=False):
        self.accelerator: Accelerator = Accelerator(mixed_precision = mixed_precision, cpu=cpu)

    def enter(self, learner: "Learner"):
        learner.accelerator = self.accelerator
        learner.device = self.accelerator.device

    def before_fit(self, learner: "Learner"):
        (
            learner.model,
            learner.optimizer,
            learner.scheduler,
            learner.dltrain,
        ) = learner.accelerator.prepare(
            learner.model,
            learner.optimizer,
            learner.scheduler,
            learner.dltrain,
        )
