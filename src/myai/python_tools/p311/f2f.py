from collections import abc
import typing

def func2func(wrapper: abc.Callable, *args, **kwargs):
    """Copies the signature from one function to another. Works with VSCode autocomplete."""

    def decorator(func: abc.Callable) -> abc.Callable:
        func.__doc__ = wrapper.__doc__
        return func

    return decorator

func2method = func2func
method2method = func2func
method2method_return_override = func2func
method2func = func2func
method2func_return_override = func2func
