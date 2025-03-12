import torch
from torch.optim import Optimizer

class AINO(Optimizer):
    """
    Absolute Integral Normalized Optimizer.

    divides sum of all past gradients by sum of absolute values of all past gradients, currently no decay!

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        eps (float, optional): Term to prevent division by zero (default: 1e-8).

    TODO add decay
    """

    def __init__(self, params, lr=1e-3, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['integral'] = torch.zeros_like(p.data)
                self.state[p]['abs_integral'] = torch.zeros_like(p.data)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Update integrals
                state['integral'].add_(grad)
                state['abs_integral'].add_(torch.abs(grad))

                # Compute adaptive update
                denom = state['abs_integral'].add(eps)
                update = state['integral'].div(denom)

                # Apply update
                p.data.add_(-lr * update)

        return loss