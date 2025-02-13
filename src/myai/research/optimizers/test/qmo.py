import torch
from torch.optim import Optimizer

@torch.no_grad
def cubic_root_nr(a, b, c, d, num_iter=10, eps=1e-8):
    """
    Newton-Raphson solver for real roots of a*x^3 + b*x^2 + c*x + d = 0.
    Prioritizes roots corresponding to minima (positive second derivative).
    """
    # Initial guess: prioritize linear term solution with fallback
    x_linear = -d / (c + eps)
    x = x_linear

    # Track best root based on second derivative (for minima)
    best_x = x
    best_fpp = float('inf')  # Initialize with large value

    for _ in range(num_iter):
        # Newton-Raphson update
        f = a * x**3 + b * x**2 + c * x + d
        fp = 3 * a * x**2 + 2 * b * x + c
        x = x - f / (fp + eps * torch.sign(fp))

        # Compute second derivative (f'') at current x
        fpp = 6 * a * x + 2 * b
        fpp_mean = fpp.mean()
        if 0 < fpp_mean < best_fpp:  # Prefer minima
            best_x = x
            best_fpp = fpp_mean

    return best_x

class QMO(Optimizer):
    """
    Quartic Momentum Optimizer.

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate (default: 1e-3)
        betas (tuple[float]): coefficients for moment estimation
            (beta1, beta2, beta3, beta4) (default: (0.9, 0.99, 0.999, 0.9999))
        eps (float): term added to denominators for numerical stability
            (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99, 0.999, 0.9999),
                 eps=1e-8, grad_clip=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, grad_clip=grad_clip)
        super().__init__(params, defaults)

        # Initialize moments and step counters
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['m1'] = torch.zeros_like(p.data)
                state['m2'] = torch.zeros_like(p.data)
                state['m3'] = torch.zeros_like(p.data)
                state['m4'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2, beta3, beta4 = group['betas']
            eps = group['eps']
            clip = group['grad_clip']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                state['step'] += 1
                t = state['step']

                # clipping
                if clip > 0:
                    grad = grad.clamp(-clip, clip)

                # Update moments with EMA
                m1, m2, m3, m4 = state['m1'], state['m2'], state['m3'], state['m4']
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                m2.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
                m3.mul_(beta3).add_(grad.pow(3), alpha=1 - beta3)
                m4.mul_(beta4).add_(grad.pow(4), alpha=1 - beta4)

                # Bias correction
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                bc3 = 1 - beta3 ** t
                bc4 = 1 - beta4 ** t

                m1_hat = m1 / bc1
                m2_hat = m2 / bc2
                m3_hat = m3 / bc3
                m4_hat = m4 / bc4

                # Construct cubic equation coefficients (derivative of quartic model)
                a = m4_hat / 6.0  # Quartic term: (1/24)Δ^4 → derivative (1/6)Δ^3
                b = m3_hat / 2.0  # Cubic term: (1/6)Δ^3 → derivative (1/2)Δ^2
                c = m2_hat        # Quadratic term: (1/2)Δ^2 → derivative Δ
                d = m1_hat        # Linear term: Δ → derivative 1

                # Solve cubic equation for optimal step Δ
                delta = cubic_root_nr(a, b, c, d, eps=eps)

                # Apply parameter update
                p.data.add_(delta, alpha=lr)

        return loss