from collections.abc import Mapping, Sequence

import numpy as np
import torch

from ..python_tools import round_significant

TAB = '    '

def _r3(x):
    if isinstance(x, (int,bool,str)): return x
    return round_significant(x, 3)

def _describe(x, level: int):
    ident = TAB * level

    if isinstance(x, (int, float)):
        return f'{ident}{type(x).__name__}({_r3(x)})'

    if isinstance(x, str):
        if len(x) < 100:
            return f'{ident}{type(x).__name__}("{x}")'
        return f'{ident}{type(x).__name__}("{x[:100]}"...)'

    if isinstance(x, torch.Tensor):
        if x.numel() <= 1:
            return f'{ident}{str(x)[:-1]}, dtype={x.dtype}, device={x.device})'

        if x.is_floating_point():
            return f'{ident}Tensor(shape={tuple(x.shape)}, mean={_r3(x.mean().item())}, std={_r3(x.std().item())}, range=({_r3(x.min().item())}, {_r3(x.max().item())}), dtype={x.dtype}, device={x.device})'

        return f'{ident}Tensor(shape={tuple(x.shape)}, range=({x.min().item()}, {x.max().item()}), dtype={x.dtype}, device={x.device})'

    if isinstance(x, np.ndarray):
        if x.size <= 1:
            return f'{ident}ndarray({str(x)}, dtype={x.dtype})'

        if x.dtype.kind == 'f':
            return f'{ident}ndarray(shape={x.shape}), mean={_r3(x.mean().item())}, std={_r3(x.std().item())}, range=({_r3(x.min().item())}, {_r3(x.max().item())}), dtype={x.dtype})'

        if x.dtype.kind == 'i':
            return f'{ident}ndarray(shape={x.shape}), range=({x.min().item()}, {x.max().item()}), dtype={x.dtype})'

        return f'{ident}ndarray(shape={x.shape}), dtype={x.dtype})'

    if isinstance(x, Sequence):
        if len(x) > 10:
            return f'{ident}{type(x).__name__}[{type[x[0]]}](len={len(x)})'

        inner = "\n".join(_describe(v, level+1) for v in x)
        return f'{ident}{type(x).__name__}(\n{inner}\n{ident})'

    if isinstance(x, Mapping):
        if len(x) > 10:
            return f'{ident}{type(x).__name__}(len={len(x)})'

        inner_d = {k: _describe(v, level+1) for k,v in x.items()}
        inner = ''
        for k,v in inner_d.items():
            inner = f'{inner}{ident}{TAB}{k} = {v[len(ident+TAB):]}\n'
        return f'{ident}{type(x).__name__}(\n{inner}{ident})'

    return f'{ident}{type(x).__name__}'


def describe(x):
    return _describe(x, 0)