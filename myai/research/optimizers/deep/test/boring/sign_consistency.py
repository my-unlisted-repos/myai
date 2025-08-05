import torch
from torch.optim import Optimizer

class SignConsistency(Optimizer):
    """
    another sign conssistency attempt

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for momentum and sign averaging (default: (0.9, 0.999))
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999)):
        defaults = dict(lr=lr, betas=betas)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("DirectionalMomentumOptimizer does not support sparse gradients")

                state = self.state[p]

                # Initialize state if necessary
                if not state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['s_avg'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(grad)

                m = state['m']
                s_avg = state['s_avg']
                prev_grad = state['prev_grad']

                state['step'] += 1

                # Update momentum
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update sign agreement average after the first step
                if state['step'] > 1:
                    sign_agree = (grad * prev_grad) > 0
                    s_t = sign_agree.float().mul_(2).sub_(1)  # Convert to 1.0 and -1.0
                    s_avg.mul_(beta2).add_(s_t, alpha=1 - beta2)

                # Effective learning rate calculation
                if state['step'] == 1:
                    effective_lr = lr * 0.5
                else:
                    effective_lr = lr * (1 + s_avg) / 2

                # Apply update
                p.data.add_(-effective_lr * m)

                # Store current gradient for next iteration
                state['prev_grad'].copy_(grad)

        return loss