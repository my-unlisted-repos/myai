import torch
from torch.optim import Optimizer

class EllipsoidOptimizer(Optimizer):
    """
    Implements a novel Ellipsoid Method-inspired optimizer for stochastic optimization.

    Key features:
    - Per-parameter tensor adaptive scaling based on gradient magnitudes.
    - Dimensionality-aware update rules to mimic ellipsoid volume reduction.
    - Automatic learning rate adjustment per parameter tensor.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): global learning rate (default: 1e-3)
        eps (float, optional): term to prevent division by zero (default: 1e-8)
        min_a (float, optional): minimum value for scaling factors (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, eps=1e-8, min_a=1e-8):
        defaults = dict(lr=lr, eps=eps, min_a=min_a)
        super().__init__(params, defaults)

        # Initialize scaling factors 'a' for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['a'] = torch.ones_like(p, requires_grad=False)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            min_a = group['min_a']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                a = state['a']

                # Get tensor dimensionality
                m = p.numel()
                if m == 0:
                    continue

                # Compute constants
                c1 = (m**2) / (m**2 - 1 + eps)
                c2 = 2.0 / (m + 1)
                c3 = 1.0 / (m + 1)

                # Compute sum term for the tensor
                sum_term = torch.sum(a * grad**2).clamp(min=eps)

                # Compute adaptive step
                step = (lr * c3) * (a * grad) / (torch.sqrt(sum_term) + eps)
                p.data.add_(-step)

                # Update scaling factors 'a'
                delta_a = c2 * (a**2 * grad**2) / sum_term
                new_a = c1 * (a - delta_a)
                new_a.clamp_(min=min_a)

                # Update state
                state['a'] = new_a

        return loss