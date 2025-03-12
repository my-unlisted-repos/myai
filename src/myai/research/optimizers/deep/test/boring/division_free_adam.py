import torch
from torch.optim import Optimizer

class DivisonFreeAdam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                # Initialize momentums
                state['m'] = torch.zeros_like(p.data)
                state['v'] = torch.zeros_like(p.data)
                # Initialize inverse approximation to 1.0 (division-free)
                state['y'] = torch.ones_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                m, v, y = state['m'], state['v'], state['y']

                # Update step counter
                state['step'] += 1

                # Update first moment (momentum)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update second moment (uncentered variance)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Division-Free Newton Step for approximating 1/(v + epsilon)
                y_new = y * (2 - (v + epsilon) * y)
                # Clip to maintain numerical stability
                y_new = torch.clamp(y_new, min=1e-10, max=1e10)
                y.copy_(y_new)

                # Update parameters
                p.data.addcmul_(m, y, value=-lr)

        return loss