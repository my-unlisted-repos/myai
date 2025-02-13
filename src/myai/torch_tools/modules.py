from collections import abc
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
