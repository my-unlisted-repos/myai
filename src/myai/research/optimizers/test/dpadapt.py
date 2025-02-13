import torch
from torch.optim import Optimizer

class DPAdapt(Optimizer):
    """a smarter dynamic discount factor (gamma) based on cosine similarity"""
    def __init__(self, params, lr=1e-3, gamma_base=0.9, beta=0.9, eps=1e-8):
        defaults = dict(lr=lr, gamma_base=gamma_base, beta=beta, eps=eps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['value_estimate'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['cos_sim_ma'] = 0.0
                state['gamma'] = gamma_base

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma_base = group['gamma_base']
            beta = group['beta']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('DPAdaptiveOptimizer does not support sparse gradients')

                state = self.state[p]
                state['step'] += 1

                prev_grad = state['prev_grad']
                value_estimate = state['value_estimate']

                # Update gamma based on cosine similarity of consecutive gradients
                if state['step'] > 1:
                    dot_product = torch.dot(prev_grad.flatten(), grad.flatten())
                    norm_prev = torch.norm(prev_grad)
                    norm_current = torch.norm(grad)

                    if norm_prev > eps and norm_current > eps:
                        cosine_sim = (dot_product / (norm_prev * norm_current)).item()
                        # Update moving average of cosine similarity
                        state['cos_sim_ma'] = beta * state['cos_sim_ma'] + (1 - beta) * cosine_sim
                        # Adjust gamma, ensuring non-negative contribution
                        adjusted_gamma = gamma_base + (1 - gamma_base) * max(state['cos_sim_ma'], 0)
                        state['gamma'] = min(adjusted_gamma, 0.999)  # Prevent gamma approaching 1.0
                    else:
                        state['gamma'] = gamma_base

                # Update value estimate: current grad + gamma * previous value_estimate
                value_estimate.mul_(state['gamma']).add_(grad)

                # Update parameters
                p.add_(-lr * value_estimate)

                # Save current gradient for next step
                state['prev_grad'].copy_(grad)

        return loss