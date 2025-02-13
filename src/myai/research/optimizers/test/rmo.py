import torch
from torch.optim.optimizer import Optimizer

class RMO(Optimizer):
    """
    Resolvent Momentum Optimizer. Goes a bit more closely to a newton step but may get wild with larger lrs.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (alpha). Default: 1e-3.
        beta (float, optional): Momentum decay factor (between 0 and 1). Default: 0.9.
        eta (float, optional): Operator scaling parameter (>=0). Default: 0.1.
    """

    def __init__(self, params, lr=1e-1, beta=0.9, eta=0.1):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta: {beta}")
        if eta < 0:
            raise ValueError(f"Invalid eta: {eta}")
        defaults = dict(lr=lr, beta=beta, eta=eta)
        super().__init__(params, defaults)

        # Initialize momentum buffers for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum'] = torch.zeros_like(p, requires_grad=False)

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
            eta = group['eta']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                m = state['momentum']

                # Update momentum buffer
                m.mul_(beta).add_(grad, alpha=1 - beta)

                # Compute numerator: eta * (grad^T grad) * m
                grad_sq = torch.dot(grad.flatten(), grad.flatten())
                numerator = eta * grad_sq * m

                # Compute denominator: 1 + eta * (m^T grad)
                m_dot_grad = torch.dot(m.flatten(), grad.flatten())
                denominator = 1 + eta * m_dot_grad

                # Compute preconditioned gradient: grad - numerator / denominator
                precond_grad = grad - numerator / denominator

                # Update parameters
                p.data.sub_(precond_grad, alpha=lr)

        return loss