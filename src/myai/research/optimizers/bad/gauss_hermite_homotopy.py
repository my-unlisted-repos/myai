# pylint:disable=signature-differs, not-callable

import torch
from torch.optim import Optimizer

class GaussHermiteHomotopy(Optimizer):
    """do those coefficients help in any way?"""
    def __init__(self, params, lr=1e-3, sigma=0.1, num_points=3):
        if num_points != 3:
            raise ValueError("This implementation currently only supports 3 quadrature points.")
        defaults = dict(lr=lr, sigma=sigma, num_points=num_points)
        super().__init__(params, defaults)
        self.points, self.weights = self._generate_quadrature()

    def _generate_quadrature(self):
        # Gauss-Hermite quadrature points and weights for n=3
        points = torch.tensor([-1.22474487, 0.0, 1.22474487], dtype=torch.float32)
        weights = torch.tensor([0.29540898, 1.1816359, 0.29540898], dtype=torch.float32)
        # Normalize weights to sum to 1
        weights /= weights.sum()
        return points, weights

    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        group = self.param_groups[0]
        lr = group['lr']
        sigma = group['sigma']

        # Check if any parameters require gradient
        params_with_grad = []
        for p in group['params']:
            if p.requires_grad:
                params_with_grad.append(p)
        if not params_with_grad:
            raise RuntimeError("No params with GRAD!!! STUPID IDIOT")

        # Save original parameters
        original_params = [p.detach().clone() for p in group['params']]

        # Generate random direction vector
        v = [torch.randn_like(p) for p in group['params']]
        # Compute norm of the direction vector
        norm = sum(torch.sum(vi ** 2) for vi in v) ** 0.5
        epsilon = 1e-8  # Avoid division by zero
        v = [vi / (norm + epsilon) for vi in v]

        total_grad = [torch.zeros_like(p) for p in group['params']]

        for x, w in zip(self.points, self.weights):
            # Perturb parameters along direction v
            for p, vi in zip(group['params'], v):
                p.add_(x * sigma, vi)

            # Zero gradients before backward pass
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            # Compute loss with perturbed parameters
            # Compute gradients
            with torch.enable_grad(): loss = closure()

            # Accumulate gradients with weight
            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    total_grad[i].add_(w * p.grad.detach())

            # Restore original parameters
            for p, orig in zip(group['params'], original_params):
                p.copy_(orig)

        # Perform parameter update
        for i, p in enumerate(group['params']):
            if p.grad is not None:
                p.add_(-lr * sigma, total_grad[i])

        return loss