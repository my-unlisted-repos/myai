import torch

def maybe_compile(fn):
    return torch.compile(fn)