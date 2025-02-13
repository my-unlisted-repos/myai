import torch
from torch.optim import Optimizer

class SMAO(Optimizer):
    """
    like lmao but smao. SignSGD but divides by EMA of absolute gradient magnitude and ites better than SignSGD.
    """

    def __init__(self, params, lr=1e-3, beta=0.9):
        """
        Initialization.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            beta (float, optional): coefficient used for computing running averages of gradient magnitudes (default: 0.9)
        """
        defaults = dict(lr=lr, beta=beta, avg_grad_magnitude=0.0)
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            loss (Tensor) or None: if closure is not None, returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    state['avg_grad_magnitude'] = torch.zeros_like(grad.mean()) # Initialize as scalar zero

                state['step'] += 1
                beta = group['beta']

                # Compute sign of gradient
                sign_grad = torch.sign(grad)

                # Compute absolute magnitude of gradient and average it
                abs_grad_mean = torch.abs(grad).mean()

                # Update moving average of absolute gradient magnitude
                state['avg_grad_magnitude'] = state['avg_grad_magnitude'] * beta + (1 - beta) * abs_grad_mean

                # Adaptive learning rate based on average gradient magnitude
                lr_adaptive = group['lr'] / (1 + state['avg_grad_magnitude'])

                # Parameter update: use sign of gradient and adaptive learning rate
                p.data.add_(sign_grad, alpha=-lr_adaptive)

        return loss