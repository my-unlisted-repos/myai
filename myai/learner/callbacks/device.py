from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from ...event_model import Callback

if TYPE_CHECKING:
    from ..learner import Learner

class Device(Callback):
    order = -20
    def __init__(self, device: torch.types.Device):
        """Callback that moves model and batches to a specified device.

        Note that batch has to be either a tensor or an iterable of tensors.

        Args:
            device (torch.device): The device to move the model to.
        """
        self.device = device

    def enter(self, learner: 'Learner'):
        if isinstance(learner.model, torch.nn.Module): learner.model = learner.model.to(learner.device)

    def before_any_batch(self, learner: 'Learner'):
        if isinstance(learner.batch, torch.Tensor):
            learner.batch = learner.batch.to(learner.device)
        else:
            learner.batch = [i.to(learner.device) for i in learner.batch]