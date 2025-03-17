import torch
from torch.optim import Optimizer

class WienerFilterOptimizer(Optimizer):
    """
    Implements the Wiener Filter-inspired optimizer for stochastic optimization.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients for computing EMAs of
            gradients (beta1), noise variance (beta2), and squared gradients (beta3)
            (default: (0.9, 0.999, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability
            (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.999), eps=1e-8):

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['noise_est'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['prev_m'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2, beta3 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization check
                if 'step' not in state:
                    raise RuntimeError("State not initialized properly")

                # Increment step counter
                state['step'] += 1
                step = state['step']

                # Retrieve parameters from state
                m, v, noise_est, prev_m = state['m'], state['v'], state['noise_est'], state['prev_m']

                # Compute residual (gradient - previous EMA)
                residual = grad - prev_m

                # Update noise estimate (EMA of residual^2)
                noise_est.mul_(beta2).addcmul_(residual, residual, value=1 - beta2)
                noise_hat = noise_est / (1 - beta2 ** step)

                # Update gradient EMA (m)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                m_hat = m / (1 - beta1 ** step)

                # Update squared gradient EMA (v)
                v.mul_(beta3).addcmul_(grad, grad, value=1 - beta3)
                v_hat = v / (1 - beta3 ** step)

                # Estimate signal and noise variances
                var_g = v_hat - m_hat.square()
                signal_est = var_g - noise_hat
                signal_est = signal_est.clamp(min=0)  # Ensure non-negativity

                # Compute Wiener gain
                gain = signal_est / (signal_est + noise_hat + eps)

                # Compute filtered gradient
                filtered_grad = gain * grad + (1 - gain) * prev_m

                # Update parameters
                p.add_(filtered_grad, alpha=-lr)

                # Save current EMA for next iteration
                state['prev_m'].copy_(m)

        return loss