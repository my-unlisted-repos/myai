from collections import abc

import torch


def make_upsample(
    m,
    ndim: int = 2,
) -> None | abc.Callable[[torch.Tensor], torch.Tensor]:
    if m is None: return None
    if isinstance(m, torch.nn.Module): return m
    if isinstance(m, type): return m(ndim)

    if ndim != 2: raise NotImplementedError(ndim)
    if isinstance(m, int): return torch.nn.Upsample(size = m)
    if isinstance(m, float): return torch.nn.Upsample(scale_factor = m)

    if isinstance(m, str) and ',' in m: m = [i.lower().strip() for i in m.split(',')]
    