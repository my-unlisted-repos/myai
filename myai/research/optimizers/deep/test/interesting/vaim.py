import torch
from torch.optim import Optimizer

class VAIM(Optimizer):
    """Variance-Adaptive Interbatch Momentum"""
    def __init__(self, params, lr=1e-3, beta=0.9, variance_ema=0.99, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, variance_ema=variance_ema, eps=eps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['update_ema'] = torch.zeros_like(p.data)
                state['update_variance'] = torch.zeros_like(p.data)
                state['momentum'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            variance_ema = group['variance_ema']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Current update candidate
                update = -lr * grad

                # Update EMA and variance
                state['update_ema'] = beta * state['update_ema'] + (1 - beta) * update
                update_diff = (update - state['update_ema'])**2
                state['update_variance'] = (
                    variance_ema * state['update_variance'] +
                    (1 - variance_ema) * update_diff
                )

                # Momentum modulation via cosine similarity
                current_direction = update.sign()
                ema_direction = state['update_ema'].sign()
                cosine_sim = (current_direction * ema_direction).sum()
                momentum = torch.sigmoid(cosine_sim)  # [0, 1]

                # Variance-adaptive learning rate scaling
                variance_scaling = 1 / (torch.sqrt(state['update_variance']) + eps)
                scaled_update = variance_scaling * update

                # Apply momentum and update
                state['momentum'] = momentum * state['momentum'] + scaled_update
                p.add_(state['momentum'] * lr)

        return loss