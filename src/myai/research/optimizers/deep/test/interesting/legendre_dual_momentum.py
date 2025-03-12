import torch
from torch.optim import Optimizer

class LegendreDualMomentum(Optimizer):
    """
    Legendre Dual Momentum Optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate. Default: 1e-3.
        beta (float, optional): Momentum factor. Default: 0.9.
        eps (float, optional): Term added to improve numerical stability. Default: 1e-8.
    """

    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

        # Initialize dual variables
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['dual_p'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                dual_p = state['dual_p']

                # Update dual variable with momentum
                dual_p.mul_(beta).add_(grad, alpha=1 - beta)

                # Compute the Legendre-inspired transformation
                update = torch.sign(dual_p) * torch.sqrt(2.0 * torch.abs(dual_p) + eps)

                # Update parameters
                p.add_(update, alpha=-lr)

        return loss