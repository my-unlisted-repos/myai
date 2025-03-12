import torch
from torch.optim import Optimizer

class ConeConstrainedMomentum(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, theta=0.5, eps=1e-8):
        """
        soft-projects momentum onto a cone, so that it doesn't diverge from gradient by more than by 1 momentum

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            beta (float, optional): Momentum decay factor. Defaults to 0.9.
            theta (float, optional): Cone constraint factor. Defaults to 0.5.
            eps (float, optional): Small value to prevent division by zero. Defaults to 1e-8.
        """
        defaults = dict(lr=lr, beta=beta, theta=theta, eps=eps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            theta = group['theta']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                m = state['momentum']

                # Update momentum with current gradient
                m.mul_(beta).add_(grad, alpha=1 - beta)

                # Compute unconstrained step
                d_unconstrained = -grad

                # Calculate difference between unconstrained step and momentum
                delta = d_unconstrained - m

                # Compute norms with epsilon to avoid division by zero
                norm_delta = torch.norm(delta) + eps
                norm_m = torch.norm(m) + eps

                # Determine the cone radius
                radius = theta * norm_m

                # Project step if outside the cone
                if norm_delta <= radius:
                    d = d_unconstrained
                else:
                    scale = radius / norm_delta
                    d = m + delta * scale

                # Apply the computed step to the parameters
                p.data.sub_(d, alpha = lr)

        return loss