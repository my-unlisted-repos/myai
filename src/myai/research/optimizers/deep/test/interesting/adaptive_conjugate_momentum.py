import torch
from torch.optim import Optimizer

class AdaptiveConjugateMomentum(Optimizer):
    """
    Adaptive Conjugate Momentum (ACM) algorithm. Goes all over the place.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): coefficient used for computing
            running averages of gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)

    """

    def __init__(self, params, lr=1e-3, beta1=0.9, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                    state['direction'] = None
                    state['prev_momentum_buffer'] = torch.zeros_like(p.data)

                momentum_buffer = state['momentum_buffer']
                direction = state['direction']
                prev_momentum_buffer = state['prev_momentum_buffer']

                beta1 = group['beta1']
                eps = group['eps']

                # Momentum update
                momentum_buffer.mul_(beta1).add_(1 - beta1, grad)

                if direction is None:
                    direction = -momentum_buffer
                else:
                    # Fletcher-Reeves style beta_cg using momentum_buffer
                    num = torch.sum(momentum_buffer * momentum_buffer)
                    den = torch.sum(prev_momentum_buffer * prev_momentum_buffer)
                    beta_cg = num / (den + eps)
                    direction = -momentum_buffer + beta_cg * direction

                p.data.add_(group['lr'], direction)

                # Store current momentum_buffer as prev_momentum_buffer for the next step
                state['prev_momentum_buffer'] = momentum_buffer.clone()
                state['momentum_buffer'] = momentum_buffer
                state['direction'] = direction

        return loss