from typing import Any, TYPE_CHECKING
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from functools import partial

import torch

from ...event_model import Callback, CancelContext, ConditionalCallback, EventModel
from ...python_tools import get__name__

if TYPE_CHECKING:
    from ..learner import Learner

class Default(Callback):
    def __init__(self, use_closure: bool = True):
        self._use_closure = use_closure

    def train(self, learner: "Learner",):
        learner.model.train()

    def eval(self, learner: "Learner"):
        learner.model.eval()

    def forward(self, learner: "Learner", inputs):
        return learner.model(inputs)

    def get_loss(self, learner: "Learner", *args):
        return learner.loss_fn(*args)

    def backward(self, learner: "Learner", loss: torch.Tensor, **kwargs):
        loss.backward(**kwargs)

    def zero_grad(self, learner: "Learner", set_to_none: bool = True):
        learner.optimizer.zero_grad(set_to_none=set_to_none)

    def optimizer_step(self, learner: "Learner", *args, **kwargs):
        learner.optimizer.step(*args, **kwargs)

    def closure(self, learner: "Learner", batch, backward: bool):
        preds = learner.forward(batch[0])
        loss = learner.get_loss(preds, batch[1])
        if backward:
            learner.zero_grad()
            learner.backward(loss)
        return loss

    def make_closure(self, learner: "Learner", batch,):
        return partial(learner.closure, batch,)

    def inference(self, learner: "Learner", inputs: torch.Tensor | Any, enable_grad = False):
        learner.eval()

        if isinstance(inputs, torch.Tensor): inputs = inputs.to(learner.device)
        elif isinstance(inputs, Sequence): inputs = [i.to(learner.device) for i in inputs]

        with torch.enable_grad() if enable_grad else torch.no_grad():
            return learner.forward(inputs)

    def _train_batch(self, learner: "Learner", batch):
        learner.train()
        learner.closure(batch)
        learner.optimizer_step()

    def _train_batch_closure(self, learner: "Learner", batch):
        learner.train()
        learner.optimizer_step(learner.make_closure(batch))

    def _test_batch(self, learner: "Learner", batch):
        learner.eval()
        with torch.no_grad():
            learner.closure(batch, backward=False) # this creates learner.preds and learner.loss

    def one_batch(self, learner: "Learner", batch, train: bool):
        if train:
            if self._use_closure: self._train_batch_closure(learner, batch)
            else: self._train_batch(learner, batch)
        else:
            self._test_batch(learner, batch)

    def one_epoch(self, learner: "Learner", dl: Iterable, train: bool):
        for learner.cur_batch, learner.batch in enumerate(dl):
            learner.one_batch(learner.batch, train = train)

    def fit(self, learner: "Learner", dltrain: Iterable, epochs_iterator: Iterable[int]):
        for learner.cur_epoch in epochs_iterator:
            learner.one_epoch(dltrain, train=True)

    def log(self, learner: "Learner", metric: str, value: Any):
        learner.logger.log(learner.num_forwards, metric, value)



class NoTarget(Callback):
    """Passes entire batch to the model, passes predictions to the loss function."""
    def closure(self, learner: "Learner", batch, backward: bool):
        preds = learner.forward(batch)
        loss = learner.get_loss(preds)
        if backward:
            learner.zero_grad()
            learner.backward(loss)
        return loss

class InputIsTarget(Callback):
    """Passes entire batch to the model, passes inputs and predictions to the loss function."""
    def closure(self, learner: "Learner", batch, backward: bool):
        preds = learner.forward(batch)
        loss = learner.get_loss(preds, batch)
        if backward:
            learner.zero_grad()
            learner.backward(loss)
        return loss

class Triplet(Callback):
    """Batch must be a length 3 sequence of anchor, positive and negative samples."""
    def closure(self, learner: "Learner", batch, backward: bool):
        preds_anchor = learner.forward(batch[0])
        preds_pos = learner.forward(batch[1])
        preds_neg = learner.forward(batch[2])
        loss = learner.get_loss(preds_anchor, preds_pos, preds_neg)
        if backward:
            learner.zero_grad()
            learner.backward(loss)
        return loss

class NoGrad(Callback):
    def __init__(self, enable = True):
        super().__init__()
        self.enable = enable

    @torch.no_grad
    def closure(self, learner: "Learner", batch, backward: bool):
        learner.zero_grad()
        preds = learner.forward(batch[0])
        loss = learner.get_loss(preds, batch[1])
        if not self.enable:
            if backward:
                learner.zero_grad()
                learner.backward(loss)
        return loss


class CustomBackwardFn(Callback):
    """function should accept learner and return gradient to be backpropped from preds"""
    def __init__(self, fn, pass_learner=False):
        self.fn = fn
        self.pass_learner = pass_learner

        self._learner_text = f'BackwardFn-{get__name__(fn)}'

    def backward(self, learner: "Learner", loss):
        if self.pass_learner: learner.preds.backward(gradient=self.fn(learner))
        else: learner.preds.backward(gradient=self.fn(learner.preds, learner.targets))