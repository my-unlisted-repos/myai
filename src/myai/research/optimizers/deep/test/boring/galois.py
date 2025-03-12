import torch
from torch.optim import Optimizer

class Galois(Optimizer):
    """uses higher-order gradient moments combined through nested radicals"""
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta = group['beta']
            lr = group['lr']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Galois does not support sparse gradients")

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m1'] = torch.zeros_like(p.data)
                    state['m2'] = torch.zeros_like(p.data)
                    state['m3'] = torch.zeros_like(p.data)
                    state['m4'] = torch.zeros_like(p.data)

                m1, m2, m3, m4 = state['m1'], state['m2'], state['m3'], state['m4']
                state['step'] += 1

                # Update moments
                m1.mul_(beta).add_(grad, alpha=1 - beta)
                m2.mul_(beta).addcmul_(grad, grad, value=1 - beta)
                m3.mul_(beta).add_(torch.abs(grad)**3, alpha=1 - beta)
                m4.mul_(beta).add_(grad**4, alpha=1 - beta)

                # Compute denominator components
                denom_part1 = torch.sqrt(m2)
                denom_part2 = torch.pow(m3 + eps, 1.0/3)
                denom_part3 = torch.pow(m4 + eps, 1.0/4)
                denom = denom_part1 + denom_part2 + denom_part3 + eps

                # Update parameters
                p.data.addcdiv_(m1, denom, value=-lr)

        return loss