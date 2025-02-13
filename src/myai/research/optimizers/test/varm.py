import torch
from torch.optim import Optimizer

class VARM(Optimizer):
    """Variance-Adjusted Robust Momentum Optimizer. Very similar to nesterov momentum."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, beta=0.1, epsilon=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta=beta, epsilon=epsilon)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('VARM does not support sparse gradients')

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['s'] = torch.zeros_like(p.data)

                m, s = state['m'], state['s']
                beta1, beta2 = group['beta1'], group['beta2']
                lr, beta, epsilon = group['lr'], group['beta'], group['epsilon']

                state['step'] += 1

                # Update moving averages
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                s.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)

                # Compute variance and clamp to avoid negative values
                var = s - m.pow(2)
                var.clamp_(min=0)

                # Calculate standard deviation with epsilon for numerical stability
                sigma = var.add(epsilon).sqrt_()

                # Compute the robust update
                update = m - beta * sigma

                # Apply update
                p.data.add_(-lr, update)

        return loss