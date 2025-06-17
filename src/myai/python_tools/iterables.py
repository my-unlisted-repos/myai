import functools
import operator
from typing import Any
from collections.abc import Iterable, Mapping, Sequence
from .types_ import SupportsGetitem

def _flatten_no_check(iterable: Iterable | Any) -> list[Any]:
    """Flatten an iterable of iterables, returns a flattened list. Note that if `iterable` is not Iterable, this will return `[iterable]`."""
    if isinstance(iterable, Iterable):
        return [a for i in iterable for a in _flatten_no_check(i)]
    return [iterable]

def flatten(iterable: Iterable) -> list[Any]:
    """Flatten an iterable of iterables, returns a flattened list. If `iterable` is not iterable, raises a TypeError."""
    if isinstance(iterable, Iterable): return [a for i in iterable for a in _flatten_no_check(i)] # type:ignore=reportUnnecessaryIsInstance
    raise TypeError(f'passed object is not an iterable, {type(iterable) = }')

def reduce_dim[X](x:Iterable[Iterable[X]]) -> list[X]: # pylint:disable=E0602
    """Reduces one level of nesting. Takes an iterable of iterables of X, and returns an iterable of X."""
    return functools.reduce(operator.iconcat, x, [])

def get0[X](x:Sequence[X]) -> X: return x[0] # pylint:disable = E0602
def get1[X](x:Sequence[X]) -> X: return x[1] # pylint:disable = E0602
def getlast[X](x:Sequence[X]) -> X: return x[-1] # pylint:disable = E0602

class BoundItemGetter[X]:
    """Stores a reference to the object and the key."""
    __slots__ = ('obj', 'key')
    def __init__(self, obj: SupportsGetitem[X], key):
        self.obj = obj
        self.key = key

    def __call__(self) -> X:
        """returns obj[index]."""
        return self.obj[self.key]

    def __repr__(self):
        return f"BoundItemGetter(obj={self.obj}, index={self.key})"
