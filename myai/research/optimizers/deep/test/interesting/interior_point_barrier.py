import torch
from torch.optim import Optimizer

class InteriorPointBarrier(Optimizer):
    """
    Interior Point Stochastic Barrier Optimizer (IPSBO)

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate. Default: 1e-3.
        mu (float, optional): Initial barrier strength. Default: 1.0.
        mu_decay (float, optional): Exponential decay rate for mu. Default: 0.99.
        epsilon (float, optional): Numerical stability term. Default: 1e-8.
        weight_decay (float, optional): L2 penalty (not traditional weight decay). Default: 0.
    """

    def __init__(self, params, lr=1e-3, mu=1.0, mu_decay=0.99, epsilon=1e-8, weight_decay=0):
        defaults = dict(lr=lr, mu=mu, mu_decay=mu_decay, epsilon=epsilon,
                        weight_decay=weight_decay, initial_mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            mu_decay = group['mu_decay']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("IPSBO does not support sparse gradients")

                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Compute barrier gradient: -2p / (p^2 + epsilon)
                barrier_grad = (2 * p) / (p.pow(2) + epsilon)

                # Update parameter: param = param - lr * (grad - mu * barrier_grad)
                p.add_(grad - mu * barrier_grad, alpha=-lr)

            # Decay mu exponentially
            group['mu'] = group['mu'] * mu_decay

        return loss