import torch
from torch.optim import Optimizer
import numpy as np

class MultiEMAOptimizer(Optimizer):
    """Optimizer using multiple squared gradient EMAs with polynomial fitting and safeguards against negative values."""

    def __init__(self, params, lr=1e-3, betas=[0.9, 0.95, 0.99], eps=1e-8):
        if len(betas) < 2:
            raise ValueError("At least two betas are required")
        self.betas = betas
        self.num_emas = len(betas)

        # Precompute effective window sizes (x) and their range
        self.x_values = [1.0 / (1 - beta) for beta in betas]
        self.min_x = min(self.x_values)
        self.max_x = max(self.x_values)

        # Vandermonde matrix for polynomial coefficients
        A = np.vander(self.x_values, N=self.num_emas, increasing=False)
        A_inv = np.linalg.inv(A)
        self.A_inv = torch.tensor(A_inv, dtype=torch.float32)

        defaults = dict(lr=lr, betas=betas, eps=eps, A_inv=self.A_inv)
        super().__init__(params, defaults)

        # Initialize state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['emas'] = [torch.zeros_like(p.data) for _ in betas]

    def step(self, closure=None):
        """Performs a parameter update with safeguards against negative EMA estimates."""
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            betas = group['betas']
            A_inv = group['A_inv']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")

                state = self.state[p]
                state['step'] += 1
                emas = state['emas']

                # Update EMAs
                for i, beta in enumerate(betas):
                    emas[i].mul_(beta).add_((1 - beta) * grad.pow(2))

                # Fit polynomial to EMAs
                ema_stack = torch.stack(emas)
                coeffs = torch.einsum('ij,j...->i...', A_inv, ema_stack)

                # Calculate polynomial minimum
                a = coeffs[0]
                b = coeffs[1]
                x_min = -b / (2.0 * a + 1e-8)

                # Clamp x_min to valid range and handle non-convex cases
                x_min_clamped = torch.clamp(x_min, min=self.min_x, max=self.max_x)
                mask = (a > 0).float()
                safe_x_min = x_min_clamped * mask + (1 - mask) * 1e8

                # Compute EMA minimum estimate
                c = coeffs[2] if self.num_emas >=3 else 0
                ema_min_computed = a * x_min_clamped.square() + b * x_min_clamped + c
                min_ema = torch.min(ema_stack, dim=0)[0]

                # Safeguard 1: Ensure EMA estimate >= minimum observed EMA
                ema_min = torch.where(a > 0, torch.maximum(ema_min_computed, min_ema), min_ema)

                # Safeguard 2: Clamp to epsilon to avoid sqrt(0)
                ema_min = torch.clamp(ema_min, min=eps)

                # Update parameters
                denom = ema_min.sqrt().add_(eps)
                p.data.addcdiv_(grad, denom, value=-lr)