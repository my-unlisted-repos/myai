import torch
from torch.optim import Optimizer

class DiagQuasiNewton(Optimizer):
    """
    Maintains exponential moving average of diagonal Hessian approximation using parameter/gradient changes.

    I tested on booth so far it works well but goes unstable after it converges, which means it works.
    """

    def __init__(self, params, lr=1e-3, beta=0.9, epsilon=1e-6):
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon)
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['prev_param'] = torch.zeros_like(p.data)
                state['prev_grad'] = torch.zeros_like(p.data)
                state['hessian_diag'] = torch.ones_like(p.data)

    def step(self, closure=None):
        """
        Performs a single optimization step
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta = group['beta']
            lr = group['lr']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                state['step'] += 1

                # Retrieve stored values
                prev_param = state['prev_param']
                prev_grad = state['prev_grad']
                D = state['hessian_diag']

                if state['step'] > 1:
                    # Compute parameter and gradient changes
                    delta_param = p.data - prev_param
                    delta_grad = grad - prev_grad

                    # Calculate curvature ratio with numerical stability
                    ratio = delta_param / (delta_grad + epsilon * torch.sign(delta_grad))

                    # Update diagonal Hessian approximation
                    D.mul_(beta).add_((1 - beta) * ratio)

                    # Apply momentum stabilization
                    D.clamp_(min=1e-4, max=1e4)

                # newton step
                p.data.add_(grad/D, alpha=-lr)

                # Store current values for next iteration
                state['prev_param'].copy_(p.data)
                state['prev_grad'].copy_(grad)

        return loss