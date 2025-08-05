from collections import abc
import typing

def func2func[**P, R](wrapper: abc.Callable[P, R]):
    """Copies the signature from one function to another. Works with VSCode autocomplete."""

    def decorator(func: abc.Callable) -> abc.Callable[P, R]:
        func.__doc__ = wrapper.__doc__
        return func

    return decorator

def func2method[**P, R](wrapper: abc.Callable[P, R]):
    """Copies the signature a function to a method. Works with VSCode autocomplete."""

    def decorator(func: abc.Callable) -> abc.Callable[typing.Concatenate[typing.Any, P], R]:
        func.__doc__ = wrapper.__doc__
        return func

    return decorator

def method2method[**P, R](wrapper: abc.Callable[typing.Concatenate[typing.Any, P], R]):
    """Copies the signature from a method to a method. Works with VSCode autocomplete."""

    # the typing.Any here is the self argumentyping.
    def decorator(func: abc.Callable[typing.Concatenate[typing.Any, typing.Any, P], R]) -> abc.Callable[typing.Concatenate[typing.Any, P], R]:
        func.__doc__ = wrapper.__doc__
        return func # type:ignore

    return decorator

def method2method_return_override[**P, R, RNew](wrapper: abc.Callable[typing.Concatenate[typing.Any, P], R], ret: type[RNew]):
    """Copies the signature from a method to a method, overrides return with the type specified in `ret`. Works with VSCode autocomplete."""

    # the typing.Any here is the self argumentyping.
    def decorator(func: abc.Callable[typing.Concatenate[typing.Any, typing.Any, P], R]) -> abc.Callable[typing.Concatenate[typing.Any, P], RNew]:
        func.__doc__ = wrapper.__doc__
        return func # type:ignore

    return decorator

def method2func[**P, R](wrapper: abc.Callable[typing.Concatenate[typing.Any, P], R]):
    """Copies the signature from a method to a function. Works with VSCode autocomplete."""

    def decorator(func: abc.Callable[typing.Concatenate[typing.Any, P], R]) -> abc.Callable[P, R]:
        func.__doc__ = wrapper.__doc__
        return func # type:ignore

    return decorator

def method2func_return_override[**P, R, RNew](wrapper: abc.Callable[typing.Concatenate[typing.Any, P], R], ret: type[RNew]):
    """Copies the signature from a method to a function. Works with VSCode autocomplete."""

    def decorator(func: abc.Callable[typing.Concatenate[typing.Any, P], R]) -> abc.Callable[P, RNew]:
        func.__doc__ = wrapper.__doc__
        return func # type:ignore

    return decorator
