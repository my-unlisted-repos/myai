import torch
from torch.optim import Optimizer

class Volterra(Optimizer):
    """
    Volterra formula based momentum. It is a very non agressive momentum but maybe it works well

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        alpha (float, optional): scaling factor for the integral term (default: 0.1)
        beta (float, optional): exponential decay factor for the integral kernel (default: 0.5)
    """

    def __init__(self, params, lr=1e-3, alpha=0.1, beta=0.5):
        defaults = dict(lr=lr, alpha=alpha, beta=beta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state if necessary
                if len(state) == 0:
                    state['S1'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['S2'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                S1 = state['S1']
                S2 = state['S2']

                # Compute the parameter update
                delta_theta = -lr * grad + alpha * (S1 + S2)

                # Update the parameters
                p.add_(delta_theta)

                # Update the state variables
                new_S1 = beta * (S1 + S2)
                new_S2 = delta_theta + beta * S2

                # Store the updated state
                state['S1'] = new_S1
                state['S2'] = new_S2

        return loss