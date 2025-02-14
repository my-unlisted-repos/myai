import torch
from torch.optim import Optimizer

class KAHO(Optimizer):
    """Kinetic Adaptive Hamiltonian Optimizer (KAHO)."""

    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):

        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            # Initialize kinetic energy average to 1.0 to avoid division by zero
            group.setdefault('K_avg', 1.0)
            # Store initial learning rate
            group['initial_lr'] = group['lr']
            # Initialize momentum buffers for each parameter
            for p in group['params']:
                self.state[p]['momentum'] = torch.zeros_like(p.data, requires_grad=False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta = group['beta']
            eps = group['eps']
            initial_lr = group['initial_lr']
            current_lr = group['lr']

            K_total = 0.0
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Update momentum
                state['momentum'].add_(-current_lr, grad)

                # Update parameters
                p.data.add_(current_lr * state['momentum'])

                # Accumulate kinetic energy: 0.5 * ||momentum||^2
                K_total += 0.5 * state['momentum'].square().sum().item()

            # Update exponential moving average of kinetic energy
            group['K_avg'] = beta * group['K_avg'] + (1 - beta) * K_total

            # Update learning rate for the next iteration
            group['lr'] = initial_lr / (group['K_avg'] ** 0.5 + eps)

        return loss