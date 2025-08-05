import torch
from torch.optim import Optimizer

class SMEO(Optimizer):
    def __init__(self, params, lr=1e-3, lambda_=0.1, beta=0.9):
        defaults = dict(lr=lr, lambda_=lambda_, beta=beta)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['y'] = torch.clone(p.data).detach()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            lambda_ = group['lambda_']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                y = state['y']
                grad = p.grad.data

                # Update the EMA of the proximal approximation z = p - lambda * grad
                z = p.data - lambda_ * grad
                y.mul_(1 - beta).add_(z, alpha=beta)

                # Compute Moreau gradient: (p - y) / lambda_
                moreau_grad = (p.data - y) / lambda_

                # Update parameters
                p.data.sub_(moreau_grad, alpha=lr)

        return loss