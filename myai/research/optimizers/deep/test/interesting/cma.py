import torch
from torch.optim import Optimizer

class CMAGradientOptimizer(Optimizer):
    """
    Implements the CMA-Gradient optimizer, blending CMA evolution paths with gradient preconditioning.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate. Default: 1e-3.
        c (float, optional): Decay rate for evolution path. Default: 0.1.
        alpha (float, optional): Decay rate for covariance matrix. Default: 0.1.
        beta (float, optional): Weight for gradient contribution to covariance. Default: 0.1.
        epsilon (float, optional): Numerical stability term. Default: 1e-8.
    """

    def __init__(self, params, lr=1e-3, c=0.1, alpha=0.1, beta=0.1, epsilon=1e-8):
        defaults = dict(lr=lr, c=c, alpha=alpha, beta=beta, epsilon=epsilon)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]

                # Initialize state if needed
                lr = group['lr']
                if len(state) == 0:
                    state['step'] = 0
                    state['evolution_path'] = torch.zeros_like(param.data)
                    state['covariance'] = torch.ones_like(param.data)
                    lr = lr*1e-4

                state['step'] += 1
                evolution_path = state['evolution_path']
                covariance = state['covariance']
                c_val = group['c']
                alpha_val = group['alpha']
                beta_val = group['beta']
                epsilon_val = group['epsilon']

                # Precondition the gradient
                preconditioned_grad = grad / (torch.sqrt(covariance) + epsilon_val)

                # Compute parameter update
                delta_param = -lr * preconditioned_grad

                # Update parameters
                param.data.add_(delta_param)

                # Update evolution path with momentum
                evolution_path.mul_(1 - c_val).add_(delta_param, alpha=c_val)

                # Adapt covariance matrix
                covariance_update = evolution_path.pow(2) + beta_val * grad.pow(2)
                covariance.mul_(1 - alpha_val).add_(covariance_update, alpha=alpha_val)

        return loss