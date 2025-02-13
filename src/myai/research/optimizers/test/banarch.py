import torch
from torch.optim import Optimizer

class BanachNeumannOptimizer(Optimizer):
    """
    Implements the Banach Neumann Optimizer (BNO), leveraging Neumann series
    approximations from Banach algebras for adaptive gradient scaling.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): exponential decay rate for gradient squared moving average (default: 0.9)
        eps (float, optional): term added to denominator to improve numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['a'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('BanachNeumannOptimizer does not support sparse gradients')

                state = self.state[p]
                a = state['a']
                state['step'] += 1

                # Update moving average of squared gradients
                a.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                # Compute denominator: 1 + lr * a + eps
                denom = a.clone().mul_(lr).add_(1).add_(eps)

                # Precondition gradient
                precond_grad = grad.div_(denom)

                # Update parameters
                p.data.add_(precond_grad, alpha=-lr)

        return loss