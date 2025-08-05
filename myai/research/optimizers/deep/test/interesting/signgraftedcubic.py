import math

import torch


class SignGraftedCubic(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta2=0.9, beta3=0.999, lambda2=1e-4, eps=1e-8, copy_sign=True):
        self.copy_sign = copy_sign
        defaults = dict(lr=lr, beta2=beta2, beta3=beta3, lambda2=lambda2, eps=eps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['v2'] = torch.zeros_like(p.data)
                state['v3'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta2 = group['beta2']
            beta3 = group['beta3']
            lambda2 = group['lambda2']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                v2 = state['v2']
                v3 = state['v3']

                # Update second and third moment estimates
                v2.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
                v3.mul_(beta3).add_(grad.pow(3), alpha=1 - beta3)

                # Compute denominator with regularization
                denom = v2 + lambda2

                # Coefficients for quadratic equation: 0.5*v3*d^2 + denom*d + grad = 0
                a = 0.5 * v3
                b = denom
                c = grad

                # Calculate discriminant
                discriminant = b.square() - 2 * a * c

                # Compute sqrt of discriminant, handling negatives
                sqrt_disc = torch.sqrt(torch.abs(discriminant) + eps) * torch.sign(discriminant + eps)

                # Possible solutions
                solution1 = (-b + sqrt_disc) / (a + eps)
                solution2 = (-b - sqrt_disc) / (a + eps)

                # Choose solution with minimal magnitude
                mask = (a.abs() < eps) | (discriminant < 0)
                fallback_step = -c / (b + eps)
                step = torch.where(mask, fallback_step, torch.where(
                    solution1.abs() < solution2.abs(), solution1, solution2))

                # Update parameters with learning rate
                if self.copy_sign: step.copysign_(p.grad)
                p.data.add_(step, alpha=-lr)

        return loss