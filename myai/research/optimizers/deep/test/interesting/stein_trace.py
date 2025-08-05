import torch
from torch.optim import Optimizer
from collections import defaultdict

class SteinTrace(Optimizer):
    """
    Implements Stein's Lemma-based Trace Estimation optimizer.

    Args:
        params (iterable): Parameters to optimize.
        lr (float): Learning rate (default: 1e-3).
        sigma (float): Noise std dev for perturbation (default: 1e-2).
        damping (float): Damping term for stability (default: 1e-5).
        beta (float): EMA decay rate for trace estimation (default: 0.9).
    """
    def __init__(self, params, lr=1e-3, sigma=1e-2, damping=1e-5, beta=0.9):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f'Invalid beta: {beta}')
        defaults = dict(lr=lr, sigma=sigma, damping=damping, beta=beta)
        super().__init__(params, defaults)
        self.state = defaultdict(dict)
        # Initialize trace EMAs
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['trace_ema'] = torch.tensor(0.0)

    def step(self, closure=None):
        if closure is None:
            raise ValueError("SteinTrace requires a closure.")

        # Save original parameters and generate noise
        original_params = []
        noises = []
        for group in self.param_groups:
            sigma = group['sigma']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                original_params.append(p.data.clone())
                noise = torch.randn_like(p.data) * sigma
                noises.append(noise)
                p.data.add_(noise)  # Perturb parameters

        # Compute gradients at perturbed parameters
        loss = closure()

        # Restore original parameters
        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                p.data.copy_(original_params[param_idx])
                param_idx += 1

        # Update parameters with scaled gradients
        noise_idx = 0
        for group in self.param_groups:
            lr = group['lr']
            sigma = group['sigma']
            damping = group['damping']
            beta = group['beta']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                grad = p.grad.data
                noise = noises[noise_idx]
                noise_idx += 1

                # Compute trace estimate and update EMA
                trace_estimate = (noise * grad).sum() / (sigma**2)
                state = self.state[p]
                trace_ema = state.get('trace_ema', torch.tensor(0.0, device=p.device))
                trace_ema = beta * trace_ema + (1 - beta) * trace_estimate
                state['trace_ema'] = trace_ema

                # Scale gradient and update
                scaled_grad = grad / (trace_ema.abs() + damping)
                p.data.add_(-lr * scaled_grad)

        return loss