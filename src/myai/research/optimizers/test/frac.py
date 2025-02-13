import torch
from torch.optim import Optimizer

class Frac(Optimizer):
    r"""I don't think this would work.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): EMA coefficient for loss (default: 0.9)
        epsilon (float, optional): term to prevent division by zero (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, beta=0.9, epsilon=1e-8):
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon)
        super().__init__(params, defaults)

        self.state['loss_ema'] = None

    @torch.no_grad()
    def step(self, closure=None, loss=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            loss (torch.Tensor, optional): The loss tensor. Required if closure is None.
        """
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        if loss is None:
            raise ValueError("Loss must be provided either via closure or loss argument")

        # Update loss EMA
        loss_tensor = loss.detach()
        if self.state['loss_ema'] is None:
            self.state['loss_ema'] = loss_tensor
        else:
            beta = self.defaults['beta']
            self.state['loss_ema'] = beta * self.state['loss_ema'] + (1 - beta) * loss_tensor

        loss_ema = self.state['loss_ema']

        for group in self.param_groups:
            lr = group['lr']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                param = p.data

                # Compute adjusted gradient
                numerator = grad * torch.abs(param) - loss_ema * torch.sign(param)
                denominator = param ** 2 + epsilon
                adjusted_grad = numerator / denominator

                # Update parameters
                p.add_(-lr * adjusted_grad)

        return loss