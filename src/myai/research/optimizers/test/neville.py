import torch
from torch.optim import Optimizer

class Neville(Optimizer):
    """Implements Neville's algorithm-based optimizer.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        m (int, optional): window size for gradient extrapolation (default: 3)
    """

    def __init__(self, params, lr=1e-3, m=3):
        if m < 1 or m > 4:
            raise ValueError("m must be between 1 and 4")
        # Precomputed and normalized coefficients for m up to 4
        coefficients = {
            1: [1.0],
            2: [-1.0/3, 2.0/3],
            3: [1.0/7, -3.0/7, 3.0/7],
            4: [-1.0/15, 4.0/15, -6.0/15, 4.0/15]
        }
        self.coefficients = coefficients
        if m not in self.coefficients:
            raise ValueError(f"Unsupported m: {m}. Choose from 1-4.")

        defaults = dict(lr=lr, m=m)
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['grad_buffer'] = []

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            m = group['m']
            coeffs_dict = self.coefficients[m]

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Update gradient buffer
                if 'grad_buffer' not in state:
                    state['grad_buffer'] = []
                buffer = state['grad_buffer']

                # Clone gradient to avoid modification
                buffer.append(grad.clone())
                # Trim buffer to max size m
                if len(buffer) > m:
                    buffer.pop(0)

                k = len(buffer)
                if k == 0:
                    continue  # No gradients to use

                # Get coefficients for current buffer size (use k up to m)
                coeffs = self.coefficients.get(k, [1.0])
                if len(coeffs) != k:
                    coeffs = self.coefficients[len(coeffs)]

                # Compute extrapolated gradient
                extrap_grad = torch.zeros_like(grad)
                for i in range(len(coeffs)):
                    extrap_grad += coeffs[i] * buffer[i]

                # Update parameters
                p.data.add_(-lr * extrap_grad)

        return loss