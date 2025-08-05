import copy
from collections import abc
from typing import Any

import numpy as np
import torch


def is_container(mod:torch.nn.Module):
    """Returns True if the module is a container"""
    if len(list(mod.children())) == 0: return False # all containers have chilren
    if len(list(mod.parameters(False))) == 0 and len(list(mod.buffers(False))) == 0: return True # containers don't do anything themselves so they can't have parameters or buffers
    return False # has children, but has params or buffers

def count_params(module:torch.nn.Module):
    """Total number of parameters in a module"""
    return sum(p.numel() for p in module.parameters())

def count_learnable_params(module:torch.nn.Module):
    """Total number of learnable parameters in a module"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def count_fixed_params(module:torch.nn.Module):
    """Total number of learnable parameters in a module"""
    return sum(p.numel() for p in module.parameters() if not p.requires_grad)

def count_buffers(module:torch.nn.Module):
    """Total number of buffers in a module"""
    return sum(b.numel() for b in module.buffers())


def replace_layers_(model:torch.nn.Module, old:type, new:abc.Callable[..., torch.nn.Module]):
    """https://www.kaggle.com/code/ankursingh12/why-use-setattr-to-replace-pytorch-layers"""
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers_(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new())

def replace_conv_(model:torch.nn.Module, old:type, new:type):
    """Bias always True!!!"""
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_conv_(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new(module.in_channels, module.out_channels, module.kernel_size,
                                  module.stride, module.padding, module.dilation, module.groups))

def replace_conv_transpose_(model:torch.nn.Module, old:type, new:type):
    """Bias always True!!!"""
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_conv_(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new(module.in_channels, module.out_channels, module.kernel_size,
                                  module.stride, module.padding, module.output_padding, module.groups, True, module.dilation))

def copy_state_dict(state: torch.nn.Module | dict[str, Any], device=None):
    """clones tensors and ndarrays, recursively copies dicts, deepcopies everything else, also moves to device if it is not None"""
    if isinstance(state, torch.nn.Module): state = state.state_dict()
    c = state.copy()
    for k,v in state.items():
        if isinstance(v, torch.Tensor):
            if device is not None: v = v.to(device)
            c[k] = v.clone()
        if isinstance(v, np.ndarray): c[k] = v.copy()
        elif isinstance(v, dict): c[k] = copy_state_dict(v)
        else:
            if isinstance(v, torch.nn.Module) and device is not None: v = v.to(device)
            c[k] = copy.deepcopy(v)
    return c
