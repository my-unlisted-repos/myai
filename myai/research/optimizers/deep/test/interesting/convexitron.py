import torch
from torch.optim import Optimizer

class Convexitron(Optimizer):
    """shuold be good"""
    def __init__(self, params, lr=1e-3, beta=0.99, gamma=0.1):
        defaults = dict(lr=lr, beta=beta, gamma=gamma)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['theta_avg'] = torch.clone(p.data).detach()
                state['step'] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                theta_avg = state['theta_avg']

                # Update theta_avg: EMA of parameters
                theta_avg.mul_(beta).add_(p.data, alpha=1 - beta)

                # Compute the update: lr * grad + gamma * (p.data - theta_avg)
                update = lr * grad + gamma * (p.data - theta_avg)

                # Apply update
                p.data.sub_(update)

                state['step'] += 1

        return loss