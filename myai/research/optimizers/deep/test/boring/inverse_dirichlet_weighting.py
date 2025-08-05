import torch
from torch.optim import Optimizer

class IDW(Optimizer):
    """
    Implements the Inverse Dirichlet Weighting optimization algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        epsilon (float, optional): term added to denominators to prevent division by zero (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, epsilon=1e-8):
        defaults = dict(lr=lr, epsilon=epsilon)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['alpha'] = torch.zeros_like(p.data)  # Initialize alpha to zeros

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
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                alpha = state['alpha']

                # Update alpha with squared gradients
                alpha.addcmul_(grad, grad, value=1.0)

                # Compute average alpha for the current parameter tensor
                sum_alpha = alpha.sum()
                num_elements = alpha.numel()
                average_alpha = sum_alpha / num_elements

                # Compute scaling factor: (average_alpha / (alpha + epsilon)) * lr
                scaling = (average_alpha / (alpha + epsilon)) * lr

                # Update parameters
                p.data.add_(-scaling * grad)

        return loss