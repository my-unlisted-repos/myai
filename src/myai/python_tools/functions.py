from functools import partial
import inspect
import typing as T
from collections.abc import Callable, Iterable
from collections import OrderedDict

from .iterables import flatten
from .identity import identity
from .objects import get__name__
type Composable[**P, R] = Callable[P, R] | Iterable[Callable[P, R]]

class Compose:
    """Compose multiple functions into a single function. Note that functions will be flattened."""
    __slots__ = ['functions']
    def __init__(self, *functions: Composable):
        self.functions = flatten(functions)

    def __call__(self, x):
        for t in self.functions:
            x = t(x)
        return x

    def __add__(self, other: Composable):
        return Compose(*self.functions, other)

    def __radd__(self, other: Composable):
        return Compose(other, *self.functions)

    def __str__(self):
        return f"Compose({', '.join(str(t) for t in self.functions)})"

    def __iter__(self):
        return iter(self.functions)

    def __getitem__(self, i): return self.functions[i]
    def __setitem__(self, i, v): self.functions[i] = v
    def __delitem__(self, i): del self.functions[i]


def compose(*functions: Composable) -> Callable:
    flattened = flatten(functions)
    if len(flattened) == 1: return flattened[0]
    return Compose(*flattened)

def maybe_compose(*functions: Callable | None | Iterable[Callable | None]) -> Callable:
    """Compose some functions while ignoring None, if got only None, returns identity."""
    flattened = [i for i in flatten(functions) if i is not None]
    if len(flattened) == 1: return flattened[0]
    if len(flattened) == 0: return identity
    return Compose(*flattened)

def get_full_signature[**P2](fn: Callable[P2, T.Any], *args: P2.args, **kwargs: P2.kwargs) -> OrderedDict[str, T.Any]:
    """Returns a dictionary of all keyword arguments of a function called with given args and kwargs."""
    sig = inspect.signature(fn).bind(*args, **kwargs)
    sig.apply_defaults()
    return sig.arguments

class Split:
    __slots__ = ['functions']
    def __init__(self, *functions: Composable | None):
        self.functions = [maybe_compose(i) for i in functions]

    def __call__(self, x):
        return [f(i) for f, i in zip(self.functions, x)]


def get_extra_signature[**P2](fn: Callable[P2, T.Any], *args: P2.args, **kwargs: P2.kwargs) -> OrderedDict[str, T.Any]:
    """Returns a dictionary of all keyword arguments of a function called with given args and kwargs."""
    sig = inspect.signature(fn).bind(*args, **kwargs)
    sig.apply_defaults()
    extra_args = sig.arguments.copy()
    extra_args['__constructor__'] = get__name__(fn)
    return extra_args

class _NotConstructed: pass
class SaveSignature[V]:
    def __init__[**K](self, obj: Callable[K, V], *args: K.args, **kwargs: K.kwargs): # pylint:disable=undefined-variable
        self.obj: Callable[..., V] = obj # pylint:disable=undefined-variable
        self.signature = get_full_signature(obj, *args, **kwargs)
        for k,v in self.signature.items():
            if isinstance(v, SaveSignature):
                self.signature[k] = v.resolve()
        self.constructed_obj: _NotConstructed | V = _NotConstructed() # pylint:disable=undefined-variable

    def resolve(self) -> V: # pylint:disable=undefined-variable
        if self.is_constructed: return self.constructed_obj # type:ignore

        # make sure *arg only arguments are passed as args
        full_argspec = inspect.getfullargspec(self.obj)
        args = []
        kwargs = self.signature.copy()

        if (full_argspec.varargs is not None) or (full_argspec.varkw is not None):

            # all kwargs from signature that are before varargs must be passed as args
            for k, v in kwargs.copy().items():
                if k == full_argspec.varargs:
                    args.extend(v)
                    del kwargs[k]
                elif k == full_argspec.varkw:
                    del kwargs[k]
                    kwargs.update(v)
                elif (full_argspec.varargs is not None) and (len(args) == 0):
                    args.append(v)
                    del kwargs[k]

        self.constructed_obj = self.obj(*args, **kwargs)
        return self.constructed_obj

    def extra_signature(self):
        """signature plus class name and constructor name, this resolves!"""
        constructed_obj = self.resolve()
        sig = self.signature.copy()
        sig.update({"__class__": get__name__(constructed_obj)})
        sig.update({"__constructor__": get__name__(self.obj)})
        return sig

    @property
    def is_constructed(self):
        return not isinstance(self.constructed_obj, _NotConstructed)

    @is_constructed.setter
    def is_constructed(self, v: bool):
        if v: raise ValueError
        self.constructed_obj = _NotConstructed()