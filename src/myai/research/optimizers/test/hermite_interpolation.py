import torch
from torch.optim import Optimizer

class HermiteInterpolation(Optimizer):
    def __init__(self, params, lr=1e-3, epsilon=1e-6, newton_steps=3, max_step=1.0):
        defaults = dict(lr=lr, epsilon=epsilon,
                       newton_steps=newton_steps, max_step=max_step)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['theta_prev'] = torch.zeros_like(p.data)
                state['g_prev'] = torch.zeros_like(p.data)
                state['h_prev'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['epsilon']
            newton_steps = group['newton_steps']
            max_step = group['max_step']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if state['step'] == 0:
                    # First step is gd
                    state['theta_prev'] = p.data.clone()
                    p.data.add_(grad, alpha=-lr)
                    state['g_prev'] = grad.clone()
                    state['h_prev'] = torch.zeros_like(p.data)
                    state['step'] += 1
                    continue

                theta_prev = state['theta_prev']
                g_prev = state['g_prev']
                h_prev = state['h_prev']
                theta_current = p.data

                # deltas
                delta_theta = theta_current - theta_prev
                delta_g = grad - g_prev

                # Hessian approximation (element-wise clamped division)
                denom = delta_theta + eps * torch.sign(delta_theta)
                denom = torch.where(denom.abs() < eps,
                                   eps * torch.sign(denom) + 1e-12,
                                   denom)
                h_current = delta_g / denom

                # Cubic Hermite coefficients
                delta_theta_sq = delta_theta.square() + 1e-12
                delta_theta_cu = delta_theta * delta_theta_sq

                a_num = delta_theta * (h_current + h_prev) - 2 * delta_g
                a = a_num / (delta_theta_cu + 1e-12)
                b_num = 3 * delta_g - (2 * h_prev + h_current) * delta_theta
                b = b_num / (delta_theta_sq + 1e-12)

                # Newton-Raphson
                x = torch.zeros_like(delta_theta)  # Start from zero displacement
                for _ in range(newton_steps):
                    H = a * x**3 + b * x**2 + h_prev * x + g_prev
                    H_prime = 3 * a * x**2 + 2 * b * x + h_prev
                    step = H / (H_prime + 1e-12)
                    x = torch.clamp(x - step, -max_step, max_step)

                # Update params
                new_theta = theta_prev + x
                p.data.copy_(new_theta)

                # Update state
                state['theta_prev'] = theta_current.clone()
                state['g_prev'] = grad.clone()
                state['h_prev'] = h_current
                state['step'] += 1

        return loss