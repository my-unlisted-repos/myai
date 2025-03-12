# pylint:disable=signature-differs, not-callable

import torch
from torch.optim import Optimizer

class MultiplicativeExponentialSign(Optimizer):
    """
    multiplicative update based on gradient sign. If parameter is 0 then it can never get updated.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 0.01).
        clip_value (float, optional): Gradient clipping value to prevent exponent overflow (default: 10.0).
        decay_rate (float, optional): Learning rate decay rate (default: 0.0).
    """
    def __init__(self, params, lr=0.01, clip_value=10.0, decay_rate=0.0):
        defaults = dict(lr=lr, clip_value=clip_value, decay_rate=decay_rate)
        super().__init__(params, defaults)
        self.state.setdefault('step', 0)

    @torch.no_grad
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        self.state['step'] += 1

        for group in self.param_groups:
            lr = group['lr']
            clip_value = group['clip_value']
            decay_rate = group['decay_rate']

            # Apply time-based learning rate decay
            current_lr = lr / (1 + decay_rate * self.state['step'])

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # Compute the sign of the gradient
                grad_sign = torch.sign(grad)

                # Adjust learning rate by sign and clip to prevent overflow
                adjusted_lr = grad_sign.mul_(current_lr)
                adjusted_lr.clamp_(-clip_value, clip_value)

                # Apply exponential multiplicative update
                p.mul_(torch.exp(adjusted_lr))

        return loss