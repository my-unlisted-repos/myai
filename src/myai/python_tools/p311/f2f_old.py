from collections.abc import Callable
import typing as T
# ---------------------------------------------------------------------------- #
#                           IDE type hints converters                          #
# ---------------------------------------------------------------------------- #
# the following can copy IDE type hints from one function to another, and add or remove arguments
# decorate a function with f2f_removex_addy to remove x leftmost arguments and add y new arguments to the left
# if you decorate method with another method, you'd typically use at least f2f_remove1_add1, to remove old self and add new self

_P = T.ParamSpec("_P")
_T = T.TypeVar("_T")
_S = T.TypeVar("_S")
_R_co = T.TypeVar("_R_co", covariant=True)

class FuncWithArgs1(T.Protocol, T.Generic[_P, _R_co]):
    def __get__(self, instance: T.Any, owner: type | None = None) -> Callable[_P, _R_co]:...
    def __call__(self, __: T.Any, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ... # pylint:disable=E1101,E0213 #type:ignore
class FuncWithArgs2(T.Protocol, T.Generic[_P, _R_co]):
    def __get__(self, instance: T.Any, owner: type | None = None) -> Callable[_P, _R_co]:...
    def __call__(self, __1: T.Any, __2:T.Any, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ... # pylint:disable=E1101,E0213 #type:ignore
class FuncWithArgs3(T.Protocol, T.Generic[_P, _R_co]):
    def __get__(self, instance: T.Any, owner: type | None = None) -> Callable[_P, _R_co]:...
    def __call__(self, __1: T.Any, __2:T.Any,__3:T.Any, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ... # pylint:disable=E1101,E0213 #type:ignore
class FuncWithArgs4(T.Protocol, T.Generic[_P, _R_co]):
    def __get__(self, instance: T.Any, owner: type | None = None) -> Callable[_P, _R_co]:...
    def __call__(self, __1: T.Any, __2:T.Any,__3:T.Any,__4:T.Any, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ... # pylint:disable=E1101,E0213 #type:ignore

__all__ = [
    "f2f",
    "f2f_add1",
    "f2f_add2",
    "f2f_remove1_add1",
    "func2method",
    "method2func",
    "method2method",
    "f2f_remove1",
    "f2f_remove2_add1",
    "f2f_remove3_add1",
    "f2f_remove3_add2",
    "f2f_remove4_add1",
    "f2f_remove4_add2",
]


def f2f(_: Callable[_P, _T]) -> Callable[[Callable[_P, _S]], Callable[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> Callable[_P, _S]:
        return fnc
    return _fnc

def f2f_add1(_: Callable[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs1[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs1[_P, _S]:
        return fnc # type:ignore
    return _fnc
func2method = f2f_add1

# add2 doesn't work (add1 does though for some reason)
def f2f_add2(_: Callable[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs2[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs2[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove1_add1(_: FuncWithArgs1[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs1[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs1[_P, _S]:
        return fnc # type:ignore
    return _fnc
method2method = f2f_remove1_add1

def f2f_remove1(_: FuncWithArgs1[_P, _T]) -> Callable[[Callable[_P, _S]], Callable[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> Callable[_P, _S]:
        return fnc # type:ignore
    return _fnc
method2func = f2f_remove1


def f2f_remove2_add1(_: FuncWithArgs2[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs1[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs1[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove3_add1(_: FuncWithArgs3[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs1[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs1[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove3_add2(_: FuncWithArgs3[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs2[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs2[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove4_add1(_: FuncWithArgs4[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs1[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs1[_P, _S]:
        return fnc # type:ignore
    return _fnc

def f2f_remove4_add2(_: FuncWithArgs4[_P, _T]) -> Callable[[Callable[_P, _S]], FuncWithArgs2[_P, _S]]:
    def _fnc(fnc: Callable[_P, _S]) -> FuncWithArgs2[_P, _S]:
        return fnc # type:ignore
    return _fnc