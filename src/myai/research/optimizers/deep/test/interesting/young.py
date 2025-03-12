import torch
from torch.optim import Optimizer

class Young(Optimizer):
    """Implements Young's inequality-based optimizer. Lr multiplied by mul because its horribly big.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            moving averages of gradient powers (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        p (float, optional): exponent for the conjugate pair (default: 1.5)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, p=1.5, mul=10000):

        q = p / (p - 1.0)  # Conjugate exponent

        defaults = dict(lr=lr, betas=betas, eps=eps, p=p, q=q)
        super().__init__(params, defaults)
        self.mul = mul

        # Initialize state
        for group in self.param_groups:
            for param in group['params']:
                self.state[param] = {}

    @torch.no_grad
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            p = group['p']
            q = group['q']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('YoungOptimizer does not support sparse gradients')

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(param.data)
                    state['n'] = torch.zeros_like(param.data)

                m, n = state['m'], state['n']
                state['step'] += 1
                step = state['step']

                # Update moving averages
                m.mul_(beta1).add_((1 - beta1) * torch.abs(grad)**p)
                n.mul_(beta2).add_((1 - beta2) * torch.abs(grad)**q)

                # Bias correction
                m_hat = m / (1 - beta1**step)
                n_hat = n / (1 - beta2**step)

                # Compute denominator
                denominator = (m_hat / p) + (n_hat / q) + eps

                # Update parameters
                param.data.add_(-lr*self.mul * grad / denominator)

        return loss