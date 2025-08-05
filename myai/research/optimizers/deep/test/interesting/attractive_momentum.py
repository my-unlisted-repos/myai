import torch
from torch.optim import Optimizer

class AttractiveMomentum(Optimizer):
    """
    Attractive Momentum Metaheuristic Optimizer TODO experiment with attraction coeff it can be quite high like 5

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum factor (default: 0.9)
        attraction_coeff (float, optional): coefficient scaling the attraction force towards historical best parameters (default: 0.1)
        beta1 (float, optional): EMA decay rate for loss (default: 0.9)
        beta2_high (float, optional): EMA decay rate for best_param_ema when loss improves (default: 0.1)
        beta2_low (float, optional): EMA decay rate for best_param_ema when loss does not improve (default: 0.01)
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, attraction_coeff=0.1,
                 beta1=0.9, beta2_high=0.1, beta2_low=0.01):
        defaults = dict(lr=lr, momentum=momentum, attraction_coeff=attraction_coeff,
                        beta1=beta1, beta2_high=beta2_high, beta2_low=beta2_low)
        super().__init__(params, defaults)
        self.loss_ema = None

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['velocity'] = torch.zeros_like(p.data)
                state['best_param_ema'] = p.data.clone()

    @torch.no_grad
    def step(self, closure):
        with torch.enable_grad(): loss = closure()
        if loss is None:
            raise ValueError("AMMO requires loss tensor during step(), e.g., optimizer.step(loss=loss)")

        # Update loss EMA
        current_loss = loss.item()
        if self.loss_ema is None:
            self.loss_ema = current_loss
        else:
            beta1 = self.defaults['beta1']
            self.loss_ema = beta1 * self.loss_ema + (1 - beta1) * current_loss

        # Determine beta2 based on loss improvement
        beta2 = self.defaults['beta2_high'] if current_loss < self.loss_ema else self.defaults['beta2_low']

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            attraction_coeff = group['attraction_coeff']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Update best_param_ema with adaptive beta2
                best_param_ema = state['best_param_ema']
                best_param_ema.mul_(1 - beta2).add_(p, alpha=beta2)

                # Compute attraction force towards best_param_ema
                attraction = attraction_coeff * (best_param_ema - p)

                # Update velocity with momentum, gradient, and attraction
                velocity = state['velocity']
                velocity.mul_(momentum).add_(grad).add_(attraction)

                # Update parameters
                p.sub_(velocity, alpha=lr)