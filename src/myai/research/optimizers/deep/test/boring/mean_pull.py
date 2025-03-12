import torch
from torch.optim import Optimizer

class MeanPull(Optimizer):
    """pulls parameters towards mean of all past parameters and there is a better version of this that incorporates loss and is called attractive momentum.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        gamma (float, optional): coefficient for the center-of-gravity term (default: 0.1)
    """

    def __init__(self, params, lr=1e-3, gamma=0.1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")

        defaults = dict(lr=lr, gamma=gamma)
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'] = torch.zeros_like(p.data, requires_grad=False)
                state['count'] = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                sum_ = state['sum']
                count = state['count']

                # Capture the parameter's current value before update
                old_param = p.data.clone()

                # Compute the parameter update
                if count == 0:
                    # First step: no CoG term
                    new_param = old_param - lr * grad
                else:
                    cog = sum_ / count
                    # Update includes CoG term
                    new_param = old_param - lr * grad + gamma * (cog - old_param)

                # Update the parameter
                p.data.copy_(new_param)

                # Update the state (sum and count)
                state['sum'] += old_param
                state['count'] += 1

        return loss