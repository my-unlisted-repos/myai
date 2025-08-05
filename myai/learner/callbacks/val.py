import typing as T
from collections.abc import Iterable
import torch

from ...event_model import ConditionalCallback

if T.TYPE_CHECKING:
    from ..learner import Learner


class TestEpoch(ConditionalCallback):
    """Use `.c_interval('before_train_step', 1024).c_on('after_fit').`"""
    order = 10
    # needs to run after metric evaluations
    def __init__(self, dl: Iterable):
        super().__init__()
        self.dl = dl

        self.accelerated = False

    def __call__(self, learner: "Learner"):
        if not self.accelerated:
            if learner.accelerator is not None:
                self.dl = learner.accelerator.prepare(self.dl)
                self.accelerated = True

        prev_dl = learner.dl
        prev_batch = learner.batch
        prev_status = learner.status

        if self.dl is None: raise ValueError('dltest is None')
        learner.one_epoch(self.dl, train = False)

        learner.dl = prev_dl
        learner.batch = prev_batch
        learner.status = prev_status

        learner.train()

def test_epoch(dl: Iterable, every: int | T.Literal['epoch'] | None, before_fit = True, after_fit = True, on_fit_exception = True):
    """Get a configured test epoch callback.

    :param dl: Test dataloader.
    :param every: Perform test epoch every n steps.
    :param before_fit: Whether to perform test epoch at the very beginning, defaults to True
    :param after_fit: Whether to perform test epoch after fit, defaults to True
    :return: TestEpoch callback.
    """
    cb = TestEpoch(dl)
    if isinstance(every, int):
        if before_fit: start = 0
        else: start = every
        cb.c_interval('before_train_step', every, start = start)
        if after_fit: cb.c_on('after_fit')
    elif every == 'epoch':
        cb.c_on('after_train_epoch')
        if before_fit: cb.c_first('before_train_step')
    else:
        if before_fit: cb.c_first('before_train_step')
        if after_fit: cb.c_on('after_fit')

    if on_fit_exception: cb.c_on('on_fit_exception')
    return cb