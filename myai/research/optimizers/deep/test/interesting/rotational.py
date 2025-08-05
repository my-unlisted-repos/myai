import torch
from torch.optim import Optimizer

class RotationalGD(Optimizer):
    """Rotational Gradient Descent optimizer.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate for the gradient term (default: 0.01).
        beta (float): Coefficient for the rotational term (default: 0.001).
    """
    def __init__(self, params, lr=0.01, beta=0.001):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta < 0.0:
            raise ValueError(f"Invalid beta: {beta}")
        defaults = dict(lr=lr, beta=beta)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                grad_flat = grad.view(-1)
                n = grad_flat.size(0)

                if n >= 2:
                    p_floor = n // 2 * 2
                    rotated_grad = torch.zeros_like(grad_flat)

                    rotated_grad[0:p_floor:2] = -grad_flat[1:p_floor:2]
                    rotated_grad[1:p_floor:2] = grad_flat[0:p_floor:2]

                    update = lr * grad_flat + beta * rotated_grad
                    p.add_(-update.view_as(p))
                else:
                    p.add_(-lr * grad)

        return loss