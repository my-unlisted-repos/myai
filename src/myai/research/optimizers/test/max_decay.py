import torch
from torch.optim import Optimizer

class MaxDecay(Optimizer):
    """
    scales by decaying maximum.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        gamma (float, optional): decay factor for max buffer (default: 0.9)
        eps (float, optional): term added to denominator to improve numerical stability (default: 1e-8)

    Tuning
        Large learning rate like 1
    """

    def __init__(self, params, lr=1e-1, gamma=0.9, eps=1e-8):
        defaults = dict(lr=lr, gamma=gamma, eps=eps)
        super().__init__(params, defaults)

        # Initialize max buffers for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['max_buffer'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('TropicalAdaptiveMax does not support sparse gradients')

                state = self.state[p]
                buffer = state['max_buffer']

                # Update the max buffer: buffer = max(gamma * buffer, |grad|)
                buffer.mul_(gamma)
                torch.maximum(buffer, torch.abs(grad), out=buffer)

                # Update parameters: p = p - lr * grad / (buffer + eps)
                p.addcdiv_(grad, buffer.add(eps), value=-lr)

        return loss