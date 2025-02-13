# pylint:disable=signature-differs, not-callable
import math

import torch
from torch.optim import Optimizer


class CosineConsistency(Optimizer):
    """directional consistency attempt"""
    def __init__(self, params, lr=1e-3, growth_rate=0.1, decay=0.99, min_strength=0.1, max_strength=10.0):
        defaults = dict(lr=lr, growth_rate=growth_rate, decay=decay,
                        min_strength=min_strength, max_strength=max_strength)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['previous_grad'] = torch.zeros_like(p)
                state['consistency'] = torch.ones_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            growth_rate = group['growth_rate']
            decay = group['decay']
            min_str = group['min_strength']
            max_str = group['max_strength']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                current_grad = p.grad
                prev_grad = state['previous_grad']
                consistency = state['consistency']

                # Normalize gradients for directional comparison
                current_norm = torch.norm(current_grad)
                prev_norm = torch.norm(prev_grad)

                if current_norm > 0 and prev_norm > 0:
                    cos_sim = torch.dot(current_grad.flatten(), prev_grad.flatten()) / (current_norm * prev_norm)
                else:
                    cos_sim = torch.tensor(0.0)

                # Update directional consistency
                consistency.mul_(1 + growth_rate * cos_sim)
                consistency.mul_(decay)
                consistency.clamp_(min=min_str, max=max_str)

                # Update parameters
                p.sub_(lr * consistency * current_grad)

                # Store current gradient for next comparison
                state['previous_grad'].copy_(current_grad)
        return loss