import torch
from torch.optim import Optimizer

class Verlet(Optimizer):
    """
    Implements Stochastic Verlet Optimizer with Damping.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        damping (float, optional): damping coefficient to reduce oscillations (default: 0.1)
        init_step_size (float, optional): initial step size for the first SGD step (default: 1e-3)
    """

    def __init__(self, params, lr=1e-3, damping=0.1, init_step_size=1e-3):

        defaults = dict(lr=lr, damping=damping, init_step_size=init_step_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            damping = group['damping']
            init_step_size = group['init_step_size']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # Initialize state if first run
                if len(state) == 0:
                    # First step: perform SGD initialization
                    state['previous_param'] = p.data.clone()
                    # Apply initial SGD step
                    p.data.add_(-init_step_size, grad)
                else:
                    prev_param = state['previous_param']
                    current_param = p.data

                    # Compute damped Verlet update
                    new_param = (2 - damping) * current_param - (1 - damping) * prev_param - lr * grad

                    # Update previous_param to current_param before overwriting
                    state['previous_param'] = current_param.clone()

                    # Apply new parameters
                    p.data.copy_(new_param)

        return loss