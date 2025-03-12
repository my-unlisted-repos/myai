from collections import deque

import torch
from torch.optim import Optimizer


class ConvexHullGradientRange(Optimizer):
    """
    Convex Hull Gradient Range Optimizer

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        k (int): Window size for gradient history (default: 5).
        epsilon (float): Small value to prevent division by zero (default: 1e-8).
    """

    def __init__(self, params, lr=1e-3, k=5, epsilon=1e-8):
        if k < 1:
            raise ValueError(f"Invalid window size k: {k}")
        defaults = dict(lr=lr, k=k, epsilon=epsilon)
        super().__init__(params, defaults)

        # Initialize gradient history buffer for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['grad_buffer'] = deque(maxlen=group['k'])
                # Initialize buffer with zeros to match parameter shape
                for _ in range(group['k']):
                    state['grad_buffer'].append(torch.zeros_like(p.data))

    @torch.no_grad
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            epsilon = group['epsilon']
            k = group['k']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Update gradient buffer with current gradient
                state['grad_buffer'].append(grad.clone())

                # Stack gradients in buffer to a tensor
                grad_tensor = torch.stack(list(state['grad_buffer']))

                # Compute element-wise min and max over the window
                min_grad, _ = grad_tensor.min(dim=0)
                max_grad, _ = grad_tensor.max(dim=0)

                # Compute range and denominator
                range_grad = max_grad - min_grad
                denominator = range_grad.add(epsilon)

                # Update parameters: p = p - lr * grad / denominator
                p.addcdiv_(grad, denominator, value=-lr, )

        return loss