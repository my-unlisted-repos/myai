import functools
import operator
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

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
