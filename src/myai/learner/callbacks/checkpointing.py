
import os
import shutil
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import torch
from ...python_tools import get_all_files
from ...event_model import Callback, ConditionalCallback

if TYPE_CHECKING:
    from ..learner import Learner


def _clean(learner: "Learner", root):
    learner_dir = learner.get_learner_dir(root)
    for r, dirs, files in os.walk(learner_dir):
        for dir in dirs:
            dir_full = os.path.join(r, dir)
            if len(get_all_files(dir_full)) == 0:
                shutil.rmtree(dir_full)


class _ConditionMetricIncrease:
    def __init__(self, metric: str):
        self.metric = metric
        self.cur = -float('inf')

    def __call__(self, m: "Learner"):
        if self.metric not in m.logger: return False
        val = m.logger.last(self.metric)
        if val > self.cur:
            self.cur = val
            return True
        return False

class _ConditionMetricDecrease:
    def __init__(self, metric: str):
        self.metric = metric
        self.cur = float('inf')

    def __call__(self, m: "Learner"):
        if self.metric not in m.logger: return False
        val = m.logger.last(self.metric)
        if val < self.cur:
            self.cur = val
            return True
        return False

class Checkpoint(ConditionalCallback):
    """save checkpoints on conditions, e.g. `after_test_epoch`.

    Args:
        state_dict (bool, optional): whether to save state dict. Defaults to True.
        logger (bool, optional): whether to save logger. Defaults to True.
        info (bool, optional): whether to save info. Defaults to True.
        text (bool, optional): whether to save text. Defaults to True.
        root (str, optional): runs folder. Defaults to 'runs'.
        in_epoch_dir (bool, optional):
            if True, saves checkpoints in current epoch folder, otherwise saves them in root learner folder. Defaults to True.
        min_epoch (int, optional):
            first epoch that the saving starts from. Defaults to True.
    """
    order = 1000
    def __init__(self, state_dict=True, logger = True, info = True, text = True, root = 'runs', in_epoch_dir = True, min_epoch=0):
        super().__init__()
        self.state_dict = state_dict
        self.logger = logger
        self.info = info
        self.text = text
        self.root = root
        self.in_epoch_dir = in_epoch_dir
        self.min_epoch = min_epoch

    def __call__(self, learner: 'Learner'):
        if learner.cur_epoch >= self.min_epoch:
            if self.in_epoch_dir: dir = learner.get_epoch_dir(self.root, postfix='checkpoint')
            else: dir = learner.get_learner_dir(self.root, postfix='checkpoint')

            learner.save(dir, mkdir=False, state_dict=self.state_dict, logger=self.logger, info=self.info, text = self.text)

    def c_increased(self, event: str, metric: str):
        """Run this callback every `every` time event happens."""
        if event not in self.events: self.events[event] = []
        self.events[event].append(_ConditionMetricIncrease(metric)) # type:ignore
        return self

    def c_decreased(self, event: str, metric: str):
        """Run this callback every `every` time event happens."""
        if event not in self.events: self.events[event] = []
        self.events[event].append(_ConditionMetricDecrease(metric)) # type:ignore
        return self

inf = float('inf')


def _improved(old, new, higher_is_better):
    if higher_is_better: return new > old
    return new < old

class CheckpointBest(Callback):
    order = 1001

    def __init__(
        self,
        metrics: str | Sequence[str],
        higher_is_better: bool | Sequence[bool],
        state_dict=True,
        logger=True,
        info=True,
        text=True,
        root="runs",
        in_epoch_dir=True,
        min_epoch = 0,
        cleanup=True,
    ):
        super().__init__()
        if isinstance(metrics, str): metrics = [metrics]
        self.metrics = metrics
        if isinstance(higher_is_better, bool): higher_is_better = [higher_is_better]
        self.higher_is_better = higher_is_better

        self.best_values = {m:(-inf if h else inf) for m,h in zip(metrics, higher_is_better)}
        self.last_dirs: dict[str, str | None] = {m: None for m in metrics}
        self.all_dirs = set()

        self.state_dict = state_dict
        self.logger = logger
        self.info = info
        self.text = text
        self.root = root
        self.in_epoch_dir = in_epoch_dir
        self.min_epoch = min_epoch
        self.cleanup = cleanup

    def _get_dir(self, learner: "Learner"):
        if self.in_epoch_dir: dir = learner.get_epoch_dir(self.root, postfix='checkpoint')
        else: dir = learner.get_learner_dir(self.root, postfix='checkpoint')
        return dir

    def _save(self, learner: "Learner", dir):
        learner.save(dir, mkdir=False, state_dict=self.state_dict, logger=self.logger, info=self.info, text = self.text)

    def after_test_epoch(self, learner: 'Learner'):
        if learner.cur_epoch >= self.min_epoch:
            for m,h in zip(self.metrics, self.higher_is_better):
                best_value = self.best_values[m]
                new_value = learner.logger.last(m)

                # check if value improved
                if _improved(best_value, new_value, h):
                    self.best_values[m] = new_value

                    # save new last dir and save checkpoint if not saved by other metric
                    dir = self._get_dir(learner)
                    self.last_dirs[m] = dir
                    if len(get_all_files(dir)) == 0:
                        self.all_dirs.add(dir) # if dir created by other callback, it won't be saved to all_dirs so it won't be cleaned up
                        self._save(learner, dir)


            # clean up dirs that are not last dirs of any metric
            used_dirs = set(self.last_dirs.values())
            for dir in self.all_dirs.copy():
                if dir not in used_dirs:
                    if os.path.isdir(dir):
                        shutil.rmtree(dir)
                    self.all_dirs.remove(dir)

        if self.cleanup: _clean(learner, self.root)


class Cleanup(Callback):
    """Cleans empty directories"""
    order = 10000
    def __init__(self, root = 'runs'):
        super().__init__()
        self.root = root


    def after_test_epoch(self, learner: 'Learner'): _clean(learner, self.root)
    def after_fit(self, learner: 'Learner'): _clean(learner, self.root)
    def on_fit_exception(self, learner: 'Learner'): _clean(learner, self.root)