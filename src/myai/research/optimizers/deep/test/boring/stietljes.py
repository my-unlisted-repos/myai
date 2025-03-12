import torch
from torch.optim import Optimizer

class Stieltjes(Optimizer):
    """Implements the Stieltjes algorithm optimizer with three-term recurrence for second moments.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): exponential decay rate for the first moment estimates (default: 0.9)
        beta2 (float, optional): exponential decay rate for the second moment estimates (default: 0.8)
        beta3 (float, optional): second exponential decay rate for the second moment estimates (default: 0.1)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.8, beta3=0.1, eps=1e-8):
        if beta2 + beta3 >= 1.0:
            raise ValueError("Sum of beta2 and beta3 must be less than 1.0")

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Stieltjes does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_sq_prev'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, exp_avg_sq_prev = (
                    state['exp_avg'],
                    state['exp_avg_sq'],
                    state['exp_avg_sq_prev']
                )

                beta1, beta2, beta3 = group['beta1'], group['beta2'], group['beta3']
                eps = group['eps']
                lr = group['lr']
                state['step'] += 1

                # Update first moment
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                # Update second moment with three-term recurrence
                # v_t = beta2 * v_{t-1} + beta3 * v_{t-2} + (1 - beta2 - beta3) * g_t^2
                new_exp_avg_sq = exp_avg_sq.mul(beta2).add(exp_avg_sq_prev, alpha=beta3)
                new_exp_avg_sq.addcmul_(grad, grad, value=1 - beta2 - beta3)

                # Update stored second moments
                state['exp_avg_sq_prev'] = exp_avg_sq.clone()
                exp_avg_sq.copy_(new_exp_avg_sq)

                # Compute denominator and update parameters
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-lr)

        return loss