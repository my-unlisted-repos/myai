import torch
from torch.optim import Optimizer

class MittagLeffler(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.5):
        defaults = dict(lr=lr, alpha=alpha)
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1
        t = self.step_count
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p.data)
                m = state['momentum']
                # Update momentum: m = m * (alpha + t - 1)/t + grad
                m.mul_((alpha + t - 1) / t).add_(grad)
                # Update parameters
                p.data.sub_(lr * m)
        return loss