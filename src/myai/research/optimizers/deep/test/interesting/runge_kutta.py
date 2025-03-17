# pylint:disable=signature-differs, not-callable

import torch
from torch.optim import Optimizer

class RungeKutta(Optimizer):
    """
    Implements an optimizer using the fourth-order Runge-Kutta (RK4) method for ODE integration.

    Uses four function evaluations to generate an update direction that is almost SGD.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (step size) for the RK4 update.
    """

    def __init__(self, params, lr=1e-3):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        """
        Performs a single optimization step using the RK4 method.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            raise ValueError("Closure required for RK4Optimizer, e.g., loss.backward() is called within closure.")

        # group = self.param_groups[0]
        lr = self.defaults['lr']

        # Initial parameters and loss evaluation to compute k1
        with torch.enable_grad(): loss = closure()
        params = [p for g in self.param_groups for p in g['params']]

        # Capture k1 = -gradient (current parameters)
        k1 = [-p.grad.clone() for p in params]
        original_params = [p.data.clone() for p in params]

        # Compute k2 (parameters + lr/2 * k1)
        for p, k in zip(params, k1):
            p.data.add_(other=k, alpha=lr/2)
        with torch.enable_grad(): loss = closure()
        k2 = [-p.grad.clone() for p in params]

        # Restore original parameters and compute k3 (parameters + lr/2 * k2)
        for p, orig in zip(params, original_params):
            p.data.copy_(orig)
        for p, k in zip(params, k2):
            p.data.add_(other=k, alpha=lr/2)
        with torch.enable_grad(): loss = closure()
        k3 = [-p.grad.clone() for p in params]

        # Restore original parameters and compute k4 (parameters + lr * k3)
        for p, orig in zip(params, original_params):
            p.data.copy_(orig)
        for p, k in zip(params, k3):
            p.data.add_(other=k, alpha=lr)
        with torch.enable_grad(): loss = closure()
        k4 = [-p.grad.clone() for p in params]

        # Restore original parameters
        for p, orig in zip(params, original_params):
            p.data.copy_(orig)

        # Compute final RK4 update: (k1 + 2*k2 + 2*k3 + k4) / 6
        with torch.no_grad():
            for p, k1_, k2_, k3_, k4_ in zip(params, k1, k2, k3, k4):
                update = (k1_ + 2*k2_ + 2*k3_ + k4_) / 6
                p.data.add_(other=update, alpha=lr)

        return loss