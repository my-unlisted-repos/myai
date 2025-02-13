
import typing as T
from collections.abc import Callable, Sequence
from torch import nn
import torch

__all__ = [
    'FuncModule',
    'func_to_named_module',
    'ensure_module',
]

class FuncModule(nn.Module):
    def __init__(self, func:Callable):
        """Wrap `func` into a torch.nn.Module.

        Args:
            func (Callable): _description_
        """
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def func_to_named_module(func:Callable, name:str | None = None) -> nn.Module:
    """Wrap `func` into a torch.nn.Module, except the module will have the same name as the function (or `name` if it isn't `None`).

    Args:
        func (Callable): The function to convert.
        name (str): Optional name of the module, if not specified then the name will be the name of the function.

    Returns:
        nn.Module: The named module.
    """
    if name is None: name = func.__name__ if hasattr(func, '__name__') else func.__class__.__name__
    name = ''.join([i for i in name if i.isalnum() or i == '_'])
    cls = type(name, (FuncModule,), {})
    return cls(func)


def ensure_module(x, named=True) -> nn.Module:
    if isinstance(x, nn.Module): return x
    elif isinstance(x, Callable):
        if named: return func_to_named_module(x)
        else: return FuncModule(x)
    else: raise TypeError(f"Can't convert {x} to module")

