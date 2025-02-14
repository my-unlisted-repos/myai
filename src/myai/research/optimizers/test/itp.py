import torch
from torch.optim import Optimizer

class ITPO(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.5, beta=0.9, gamma=0.999,
                 eta=1.0, epsilon=1e-8, max_step_ratio=0.1, min_step_eps=1e-3):
        defaults = dict(lr=lr, alpha=alpha, beta=beta, gamma=gamma, eta=eta,
                        epsilon=epsilon, max_step_ratio=max_step_ratio,
                        min_step_eps=min_step_eps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['m'] = torch.zeros_like(p.data)
                state['v'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ITPOptimizer does not support sparse gradients')

                state = self.state[p]
                m, v = state['m'], state['v']
                lr = group['lr']
                alpha = group['alpha']
                beta = group['beta']
                gamma = group['gamma']
                eta = group['eta']
                epsilon = group['epsilon']
                max_step_ratio = group['max_step_ratio']
                min_step_eps = group['min_step_eps']

                state['step'] += 1

                # Update momentum (exponential moving average)
                m.mul_(beta).add_(1 - beta, grad)
                # Update squared gradient norm estimate
                v.mul_(gamma).addcmul_(1 - gamma, grad, grad)

                # Interpolate between momentum and current gradient
                interpolated_step = alpha * m + (1 - alpha) * grad

                # Truncate based on adaptive threshold (eta * sqrt(v))
                denom = v.sqrt().add_(epsilon)
                threshold = eta * denom
                truncated_step = torch.clamp(interpolated_step, -threshold, threshold)

                # Project step relative to parameter magnitude
                param_current = p.data
                param_abs = torch.abs(param_current)
                # Ensure minimum step size to avoid zero updates
                max_step = torch.clamp(max_step_ratio * param_abs, min=min_step_eps)
                projected_step = torch.sign(truncated_step) * torch.min(
                    torch.abs(truncated_step), max_step)

                # Apply the final projected step
                p.data.add_(-lr * projected_step)

        return loss