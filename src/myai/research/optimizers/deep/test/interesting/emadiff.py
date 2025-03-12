import torch
from torch.optim import Optimizer

class EMADiff(Optimizer):
    """
    uses EMA of gradient differences.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): EMA decay factor for gradient differences (default: 0.9)
        epsilon (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, beta=0.9, epsilon=1e-8):
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['previous_grad'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['step'] = 0

                state['step'] += 1
                step = state['step']
                previous_grad = state['previous_grad']
                v = state['v']

                # Compute gradient difference
                delta = grad - previous_grad

                # Update EMA of absolute differences
                v.mul_(beta).add_((1 - beta) * torch.abs(delta))

                # Bias correction
                bias_correction = 1 - (beta ** step)
                v_hat = v / bias_correction

                # Adaptive learning rate
                lr_adapt = lr / (v_hat + epsilon)

                # Update parameters
                p.data.add_(-lr_adapt * grad)

                # Save current gradient for next iteration
                state['previous_grad'] = grad.clone().detach()

        return loss