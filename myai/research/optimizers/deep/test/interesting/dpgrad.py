import torch
from torch.optim.optimizer import Optimizer

class DPGrad(Optimizer):
    """
    Implements the DPGrad optimizer.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        alpha (float, optional): decay rate for gradient accumulation (default: 0.9)
        gamma (float, optional): discount factor for future gradient impact (default: 0.99)
        eps (float, optional): term added to denominator to improve numerical stability (default: 1e-8)

    Tuning:
        low lr like 1e-4
    """

    def __init__(self, params, lr=1e-3, alpha=0.9, gamma=0.99, eps=1e-8):
        # if not 0.0 <= lr:
        #     raise ValueError(f"Invalid learning rate: {lr}")
        # if not 0.0 <= alpha < 1.0:
        #     raise ValueError(f"Invalid alpha value: {alpha}")
        # if not 0.0 <= gamma < 1.0:
        #     raise ValueError(f"Invalid gamma value: {gamma}")
        # if not 0.0 <= eps:
        #     raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, alpha=alpha, gamma=gamma, eps=eps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['grad_acc'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['future_val'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['prev_update'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            gamma = group['gamma']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                state['step'] += 1
                grad_acc = state['grad_acc']
                future_val = state['future_val']
                prev_update = state['prev_update']

                # Update gradient accumulation with decay
                grad_acc.mul_(alpha).add_(grad, alpha=1 - alpha)

                # Update future value estimate using Bellman-like equation
                future_val.mul_(gamma).add_(grad_acc, alpha=1 - gamma)

                # Compute dynamic programming-inspired update
                denom = future_val.abs().add(eps)
                adaptive_lr = lr / denom.sqrt()
                current_update = adaptive_lr * grad_acc

                # Apply update with momentum from previous step
                final_update = current_update + gamma * prev_update
                p.add_(-final_update)

                # Store current update for next step's momentum
                prev_update.copy_(final_update)

        return loss