from collections import abc as A

import torch

from .func import ensure_module


class ModuleList(torch.nn.ModuleList):
    """Module list that also accepts callables."""
    def __init__(self, modules: A.Iterable[torch.nn.Module | A.Callable] | None = None):
        super().__init__([ensure_module(i) for i in modules] if modules is not None else None)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'Forward not implemented for {self.__class__.__name__}')

class Sequential(torch.nn.Sequential):
    """Sequential module that also accepts callables."""
    def __init__(self, *args: torch.nn.Module | A.Callable):
        super().__init__(*[ensure_module(i) for i in args])