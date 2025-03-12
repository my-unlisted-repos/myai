import torch
from torch.optim import Optimizer

class EMASimplex(Optimizer):
    """Implements the Simplex Optimizer algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): EMA decay factor (default: 0.9)
        eps (float, optional): term added to denominators to improve numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

        # Initialize centroid state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['centroid'] = torch.zeros_like(p.data, requires_grad=False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
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

                grad = p.grad.data
                state = self.state[p]
                centroid = state['centroid']

                # Update centroid using EMA
                centroid.mul_(beta).add_(grad, alpha=1 - beta)

                # Compute reflection: 2*centroid - grad
                reflection = 2.0 * centroid - grad

                # Compute cosine similarity between reflection and grad
                dot_product = torch.sum(reflection * grad)
                norm_reflection = torch.norm(reflection)
                norm_grad = torch.norm(grad)
                cos_sim = dot_product / (norm_reflection * norm_grad + eps)

                # Compute adaptive blending factor
                alpha = 0.5 * (1.0 - cos_sim)

                # Compute final update direction
                update = alpha * grad + (1.0 - alpha) * reflection

                # Update parameters
                p.data.add_(-lr * update)

        return loss