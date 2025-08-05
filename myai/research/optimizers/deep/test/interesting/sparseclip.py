# pylint:disable=signature-differs, not-callable

import torch
from torch.optim import Optimizer

class SparseClip(Optimizer):
    """gsda ok LR SHOULD BE 1!!! this a. uses parameter norm to adapt lr 2. this clips update based on parameter magnitudes LEADING TO SPARSITY MYABE"""
    def __init__(self, params, lr=1., delta_mul=0.1, eps=1e-8):
        defaults = dict(lr=lr, delta_mul=delta_mul, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            delta_mul = group['delta_mul']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('ParamBasedClamp optimizer does not support sparse gradients')

                # Calculate the L2 norms for gradient and parameter
                grad_norm = torch.linalg.vector_norm(grad, 2)
                param_norm = torch.linalg.vector_norm(p, 2)

                # Effective learning rate for this parameter
                effective_lr = lr * (grad_norm / (param_norm + eps))

                # Calculate the proposed update
                delta = effective_lr * grad

                # Compute the maximum allowed delta, adding eps to avoid zero clamping
                max_delta = delta_mul * (torch.abs(p) + eps)

                # Clip the delta to the range [-max_delta, max_delta]
                delta_clipped = torch.clamp(delta, -max_delta, max_delta)

                # Apply the clipped update
                p.sub_(delta_clipped)

        return loss