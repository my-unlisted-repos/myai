import itertools
import time
import typing as T
import warnings
from collections import abc

import torch

from ...event_model import Callback, CancelContext

if T.TYPE_CHECKING:
    from ..learner import Learner


class RetainGraph(Callback):
    def __init__(self, retain_graph = None):
        self.retain_graph = retain_graph

    def enter(self, learner: 'Learner'):
        self.prev_value = learner.backward_kwargs.get('retain_graph', None)
        learner.backward_kwargs['retain_graph'] = self.retain_graph

    def exit(self, learner: 'Learner'):
        learner.backward_kwargs['retain_graph'] = self.prev_value

# create_graph
class CreateGraph(Callback):
    def __init__(self, create_graph: bool = False):
        self.create_graph = create_graph

    def enter(self, learner: 'Learner'):
        self.prev_value = learner.backward_kwargs.get('create_graph', False)
        learner.backward_kwargs['create_graph'] = self.create_graph

    def exit(self, learner: 'Learner'):
        learner.backward_kwargs['create_graph'] = self.prev_value
