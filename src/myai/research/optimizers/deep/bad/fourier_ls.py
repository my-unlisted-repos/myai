# pylint:disable=signature-differs, not-callable

import torch
import torch.optim as optim

class FourierLineSearch(optim.Optimizer):
    """uses fourier series to model the function along gradient direction, would be better to model entire function, but how?"""
    def __init__(self, params, lr=0.1, n_samples=5, n_eval_points=100):
        if n_samples < 3:
            raise ValueError("n_samples must be at least 3")
        defaults = dict(h=lr, n_samples=n_samples, n_eval_points=n_eval_points)
        super(FourierLineSearch, self).__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for FourierOptimizer")

        # Compute initial loss and gradients
        with torch.enable_grad(): initial_loss = closure()
        initial_loss = initial_loss.item()

        # Save initial parameters and gradients
        initial_params = []
        initial_grads = []
        for group in self.param_groups:
            for p in group['params']:
                initial_params.append(p.detach().clone())
                if p.grad is None:
                    initial_grads.append(None)
                else:
                    initial_grads.append(p.grad.detach().clone())

        group = self.param_groups[0]
        h = group['h']
        n_samples = group['n_samples']
        n_eval_points = group['n_eval_points']

        # Generate sample alphas (centered around 0)
        alphas = torch.linspace(-h, h, steps=n_samples)

        losses = []
        for alpha in alphas:
            # Perturb parameters along the gradient direction
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    grad = initial_grads[idx]
                    if grad is not None:
                        p.data.copy_(initial_params[idx] + alpha * grad)
                    idx += 1

            # Compute loss without gradient tracking
            with torch.no_grad():
                loss = closure(False)
            losses.append(loss.item())

        # Restore original parameters and gradients
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(initial_params[idx])
                if initial_grads[idx] is not None:
                    p.grad.data.copy_(initial_grads[idx])
                idx += 1

        # Prepare data for Fourier fitting
        alphas_tensor = alphas
        losses_tensor = torch.tensor(losses, dtype=torch.float32)

        # Design matrix: [1, cos(π/h α), sin(π/h α), cos(2π/h α)]
        A = torch.stack([
            torch.ones_like(alphas_tensor),
            torch.cos(torch.pi * alphas_tensor / h),
            torch.sin(torch.pi * alphas_tensor / h),
            torch.cos(2 * torch.pi * alphas_tensor / h)
        ], dim=1).float()

        # Solve least squares problem
        coefficients = torch.linalg.lstsq(losses_tensor.unsqueeze(1), A).solution.squeeze()
        coefficients = coefficients[:4]  # Get the first 4 coefficients

        c0, c1, d1, c2 = coefficients

        # Evaluate the Fourier model on a fine grid
        alpha_eval = torch.linspace(-h, h, steps=n_eval_points)
        model_eval = c0 + c1 * torch.cos(torch.pi * alpha_eval / h) + \
                     d1 * torch.sin(torch.pi * alpha_eval / h) + \
                     c2 * torch.cos(2 * torch.pi * alpha_eval / h)

        # Find optimal alpha
        min_idx = torch.argmin(model_eval)
        alpha_opt = alpha_eval[min_idx]

        # Update parameters with the optimal alpha
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                grad = initial_grads[idx]
                if grad is not None:
                    p.data.copy_(initial_params[idx] + alpha_opt * grad)
                idx += 1

        return torch.tensor(initial_loss, dtype=torch.float32)