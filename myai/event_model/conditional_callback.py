import typing as T
from abc import ABC, abstractmethod
from collections.abc import Callable

from .callback import Callback

if T.TYPE_CHECKING:
    from .event_model import EventModel

def _return_true(_): return True

class _ConditionFirst:
    def __init__(self, n:int):
        self.n = n
        self.step = 0
    def __call__(self, m: "EventModel"):
        self.step += 1
        if self.step <= self.n: return True
        return False

class _ConditionInterval:
    def __init__(self, every:int, start: int):
        self.every = every
        self.start = start
        self.step = 0
    def __call__(self, m: "EventModel"):
        res = False
        if self.step >= self.start and self.step % self.every == 0: res = True
        self.step += 1
        return res

class ConditionalCallback(Callback, ABC):
    """A callback that can be conditionally added to the model."""
    def __init__(self, agg_fn = any):
        self.events: "dict[str, list[Callable[[EventModel], bool]]]" = {}
        self.agg_fn = agg_fn
        self.triggered = False

    @abstractmethod
    def __call__(self, __model: "EventModel", *args, **kwargs):
        ...

    def _conditional_run(self, model: "EventModel", event:str, *args, **kwargs):
        """This gets added as a `partial(_conditional_run, event=event_name)`."""
        self.triggered = False
        if self.agg_fn(f(model) for f in self.events[event]):
            self.triggered = True
            self(model, *args, **kwargs)

    def c_on(self, event: str):
        """Run this callback every time event happens."""
        if event not in self.events: self.events[event] = []
        self.events[event].append(_return_true)
        return self

    def c_first(self, event:str, n: int = 1):
        """Run this callback first n times event happens."""
        if event not in self.events: self.events[event] = []
        self.events[event].append(_ConditionFirst(n))
        return self

    def c_interval(self, event: str, every: int, start: int = 0):
        """Run this callback every `every` time event happens."""
        if event not in self.events: self.events[event] = []
        self.events[event].append(_ConditionInterval(every, start))
        return self

    def c_fn(self, event: str, fn: "Callable[[EventModel], bool]"):
        """Run this callback if `fn(model)` returns True when event happens."""
        if event not in self.events: self.events[event] = []
        self.events[event].append(fn)
        return self