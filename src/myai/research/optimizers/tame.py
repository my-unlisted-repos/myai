# pylint:disable=signature-differs, not-callable # type:ignore

import torch
from torch.optim import Optimizer

class TAME(Optimizer):
    """now why would this have good initial convergence"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9), alpha=0.1,
                 scale=1.0, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, alpha=alpha, scale=scale, eps=eps)
        super().__init__(params, defaults)

    # def __setstate__(self, state):
    #     super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('TAME does not support sparse gradients')

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['s'] = torch.zeros_like(p.data)
                    state['c'] = torch.zeros_like(p.data)

                m, v, s, c = state['m'], state['v'], state['s'], state['c']
                beta1, beta2, beta3 = group['betas']
                alpha = group['alpha']
                scale = group['scale']
                eps = group['eps']
                lr = group['lr']
                state['step'] += 1

                # Update first moment
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update second moment
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Normalize gradient
                g_norm = grad / (torch.sqrt(v) + eps)

                # Update trigonometric moments
                s.mul_(beta3).add_(torch.sin(scale * g_norm), alpha=1 - beta3)
                c.mul_(beta3).add_(torch.cos(scale * g_norm), alpha=1 - beta3)

                # Compute amplitude
                a = torch.sqrt(s**2 + c**2) + eps

                # Update parameters
                denom = torch.sqrt(v) + alpha * a + eps
                p.data.addcdiv_(m, denom, value=-lr)

        return loss