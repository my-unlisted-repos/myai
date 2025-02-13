# pylint:disable=signature-differs, not-callable
"""DeepSeek"""
import itertools
import random
from collections.abc import Callable

import numpy as np
import torch
import torchzero as tz
from torch import nn


class FFTSGD(torch.optim.Optimizer):
    """smoothes the gradient using fft, unlike laplacian smoothing sgd this doesnt flatten gradient and applies spatially.

    very slow and horrible convergence."""
    def __init__(self, params, lr=1e-3, momentum:float=0, dampening:float=0,
                 weight_decay:float=0, nesterov=False, filter_threshold=0.1):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        filter_threshold=filter_threshold)
        if momentum < 0 or dampening < 0 or lr < 0 or weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            lr = group['lr']
            filter_threshold = group['filter_threshold']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)

                # Apply FFT to the gradient
                d_p_fft = torch.fft.fftn(d_p)
                # Apply a low-pass filter in frequency domain
                # freq = torch.fft.fftfreq(d_p.size(0))
                mask = torch.ones_like(d_p_fft)
                for i in range(d_p.dim()):
                    freq_dim = torch.fft.fftfreq(d_p.size(i), d=1.0, device=d_p_fft.device)
                    for dim in list(range(i)) + list(range(i+1, d_p.dim())): freq_dim = freq_dim.unsqueeze(dim)
                    mask = mask * (freq_dim.abs() < filter_threshold).float()
                d_p_fft = d_p_fft * mask
                # Inverse FFT to get filtered gradient
                d_p_filtered = torch.fft.ifftn(d_p_fft).real

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p_filtered).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p_filtered, alpha=1 - dampening)
                    if nesterov:
                        d_p_filtered = d_p_filtered + momentum * buf
                    else:
                        d_p_filtered = buf

                p.data.add_(d_p_filtered, alpha=-lr)

        return loss
