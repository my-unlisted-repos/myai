import torch
from torch.optim import Optimizer

class SCGMA(Optimizer):
    """
    Stochastic Conjugate Gradient with Momentum Approximation.
    """

    def __init__(self, params, lr=1e-3, ema_decay=0.9, reset_interval=10, epsilon=1e-8):
        defaults = dict(lr=lr, ema_decay=ema_decay, reset_interval=reset_interval, epsilon=epsilon)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SCGMA does not support sparse gradients')

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['ema_grad'] = torch.zeros_like(p.data)
                    state['prev_dir'] = torch.zeros_like(p.data)

                state['step'] += 1
                current_step = state['step']
                lr = group['lr']
                ema_decay = group['ema_decay']
                reset_int = group['reset_interval']
                eps = group['epsilon']

                # Update EMA and handle first step
                if current_step == 1:
                    state['ema_grad'] = grad.clone()
                    state['prev_dir'] = -grad.clone()
                    p.data.add_(state['prev_dir'], alpha=lr)
                else:
                    # Compute gradient difference
                    delta_g = grad - state['ema_grad']

                    # Compute beta using smoothed gradients
                    numerator = torch.dot(grad.flatten(), delta_g.flatten())
                    denominator = torch.dot(state['ema_grad'].flatten(), state['ema_grad'].flatten()) + eps
                    beta = numerator / denominator

                    # Reset direction periodically
                    if reset_int > 0 and (current_step % reset_int) == 0:
                        beta = 0.0

                    # Update conjugate direction
                    new_dir = -grad + beta * state['prev_dir']

                    # Parameter update
                    p.data.add_(new_dir, alpha=lr)

                    # Update EMA and store direction
                    state['ema_grad'].mul_(ema_decay).add_(grad, alpha=1 - ema_decay)
                    state['prev_dir'] = new_dir.clone()

        return loss