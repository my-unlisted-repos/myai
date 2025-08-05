import bisect
import inspect
import typing as T
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from operator import attrgetter

if T.TYPE_CHECKING:
    from .event_model import EventModel

class Callback(ABC):
    """A callback.
    All methods except `enter`, `exit`, ones already defined on EventModel, and ones starting with `_`
    will be accessible by the getattr of EventModel."""

    order: int | float = 0
    """Order of execution (lower order callbacks execute first)"""

    _learner_text: str = ''
    """Optional text to display at the end of the learner name."""

    def enter(self, __model):
        """Runs when this callback is added to the model."""
    def exit(self, __model):
        """Runs when this callback is removed from the model."""

    @contextmanager
    def context(self, __model):
        """This callback is added to the model when this context starts, and gets removed when it ends."""
        __model.add_callback(self)
        try:
            yield
        finally:
            __model.remove_callbacks(self)

def _get_valid_methods(callback:Callback):
    """Returns all methods that don't start with `_` and aren't reserved."""
    for name, method in inspect.getmembers(callback, predicate=inspect.ismethod):
        if (not name.startswith(('_', 'c_'))) and name not in {'enter', 'exit', 'context'}:
            yield name, method


class _CallbackMethod:
    """Stores a callback and one of its method, as well as the order for insorting,
    and equality defined as callback being the same for removing."""
    def __init__(self, callback: Callback, method: Callable):
        self.callback = callback
        self.method = method
        self.order = callback.order

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)

    def __eq__(self, other: "_CallbackMethod"): # type:ignore
        return self.callback is other.callback


_get_order = attrgetter('order')