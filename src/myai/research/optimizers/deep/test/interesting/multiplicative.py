import torch
from torch.optim import Optimizer

class AdaptiveMultiplicativeMomentum(Optimizer):
    """
    Adaptive Multiplicative Momentum Optimizer

    Features:
    - Multiplicative updates using exponential momentum
    - Gradient normalization for stability
    - Adaptive learning rate scaling per parameter
    - Momentum decay for escaping saddle points
    """

    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-6, clamp=20.0, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, eps=eps, clamp=clamp, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
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
                lr = group['lr']
                beta = group['beta']
                eps = group['eps']
                clamp = group['clamp']
                wd = group['weight_decay']

                # Add weight decay
                if wd != 0:
                    grad = grad.add(p.data, alpha=wd)

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)
                    state['variance'] = torch.ones_like(p.data)

                m, v = state['momentum'], state['variance']

                # Update momentum with normalized gradient
                scaled_grad = grad / (grad.norm(2) + eps)
                m.mul_(beta).add_(scaled_grad, alpha=1 - beta)

                # Adaptive learning rate scaling
                adaptive_lr = lr / (v.sqrt() + eps)
                v.mul_(beta).addcmul_(m, m, value=1 - beta)

                # Compute update components
                directional_update = m.sign() * adaptive_lr
                momentum_update = torch.exp(-directional_update * m.abs())

                # Apply clamped multiplicative update
                update_factor = torch.clamp(momentum_update, 1/clamp, clamp)
                p.div_(update_factor)

        return loss