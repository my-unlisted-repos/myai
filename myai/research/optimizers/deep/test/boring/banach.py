import torch
from torch.optim import Optimizer

class Banach(Optimizer):
    """Implements Functional Gradient Descent in Banach Spaces.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        p (float, optional): Exponent of the Banach space's L^p norm (default: 2.0).
        eps (float, optional): Term added to avoid division by zero (default: 1e-8).
    """

    def __init__(self, params, lr, p=2.0, eps=1e-8):
        if not 1.0 < p < float('inf'):
            raise ValueError(f"Invalid p: {p}. Must be in (1, inf)")
        q = p / (p - 1)
        defaults = dict(lr=lr, p=p, q=q, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            q = group['q']
            eps = group['eps']

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                # Flatten gradient to compute L^q norm
                flat_grad = grad.contiguous().view(-1)
                norm = torch.norm(flat_grad, p=q)

                # Compute scaling factor
                scaling = (norm + eps) ** (2 - q)

                # Compute transformed gradient
                transformed_grad = torch.sign(grad) * (grad.abs() + eps).pow(q - 1)
                transformed_grad.mul_(scaling)

                # Update parameters
                param.data.add_(transformed_grad, alpha=-lr)

        return loss