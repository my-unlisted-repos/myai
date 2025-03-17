import torch
from torch.optim import Optimizer

class BifurcationSGD(Optimizer):
    """basically SGD"""
    def __init__(self, params, lr=1e-4, alpha=0.1, beta=0.1, gamma=0.1,
                 threshold=1e-3, state_lr=0.01, damping=0.5, cubic_scale=0.1):
        defaults = dict(lr=lr, alpha=alpha, beta=beta, gamma=gamma,
                        threshold=threshold, state_lr=state_lr,
                        damping=damping, cubic_scale=cubic_scale)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['s'] = torch.zeros_like(p.data)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            beta = group['beta']
            gamma = group['gamma']
            threshold = group['threshold']
            state_lr = group['state_lr']
            damping = group['damping']
            cubic_scale = group['cubic_scale']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                s = state['s']

                # Bifurcation parameter with gradient magnitude conditioning
                mu = gamma * (threshold - grad.abs().clamp(max=threshold))

                # Stabilized state update with damping and cubic regularization
                s_update = (mu * s) - (damping * s) + (beta * grad) - (cubic_scale * torch.pow(s, 3))
                s_new = s + state_lr * s_update

                # Clip state to prevent extreme values (optional but recommended)
                s_new = torch.clamp(s_new, min=-10.0, max=10.0)

                state['s'] = s_new.detach()

                # Parameter update with momentum-like bifurcation term
                p.data.add_(-lr * (grad + alpha * s_new))

        return loss