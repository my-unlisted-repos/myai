# pylint:disable=signature-differs, not-callable

import math
import torch
from torch.optim import Optimizer

def lambertw(x, iterations=5):
    # Approximate Lambert W using Newton-Raphson method
    # x: input tensor
    # iterations: number of NR steps
    z = torch.zeros_like(x)
    # Initial conditions based on x's sign
    mask_neg = x < 0
    mask_pos = ~mask_neg

    # For x >= 0: initial guess log(x + 1)
    z[mask_pos] = torch.log(x[mask_pos] + 1.0)

    # For x < 0: initial guess from asymptotic expansion near -1/e
    x_neg = x[mask_neg]
    z_neg = -1.0 + torch.sqrt(2.0 * (1.0 + math.e * x_neg))
    z[mask_neg] = z_neg

    for _ in range(iterations):
        exp_z = torch.exp(z)
        numerator = z * exp_z - x
        denominator = exp_z * (z + 1.0) + 1e-8  # Prevent division by zero
        delta = numerator / denominator
        z -= delta
    return z

class LambertW(Optimizer):
    """lambert W more like labert L"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(LambertW, self).__init__(params, defaults)

    @torch.no_grad
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
                    raise RuntimeError('LambertWOptimizer does not support sparse gradients')

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Update first and second moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                denom = v_hat.sqrt().add_(group['eps'])

                # Compute argument for Lambert W
                arg = - (group['lr'] * m_hat) / denom

                # Clamp argument to domain of Lambert W (real numbers >= -1/e)
                arg_clamped = torch.clamp(arg, min=-1.0/math.e)

                # Compute Lambert W approximation
                w = lambertw(arg_clamped)

                # Update parameters
                p.data.add_(w)

        return loss