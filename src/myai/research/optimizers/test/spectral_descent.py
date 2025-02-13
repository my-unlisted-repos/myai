import torch
from torch.optim import Optimizer

class SpectralDescent(Optimizer):
    """
    uses online power iteration to estimate the dominant eigenvector of gradient covariance for adaptive scaling.

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): eigenvector moving average factor (default: 0.9)
        beta2 (float, optional): eigenvalue moving average factor (default: 0.999)
        epsilon (float, optional): numerical stability term (default: 1e-12)

    Tuning:
        Requires larger lr like 1e-1
    """

    def __init__(self, params, lr=1e-1, beta1=0.9, beta2=0.999, epsilon=1e-12):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super().__init__(params, defaults)

        # Initialize spectral states
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # Initialize dominant eigenvector estimate with random direction
                    state['v'] = torch.randn_like(p).sign()  # Rademacher initialization
                    state['v'] /= state['v'].norm() + epsilon
                    state['sigma'] = torch.zeros(1, device=p.device)  # Eigenvalue estimate
                    state['step'] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                state['step'] += 1
                v = state['v']
                sigma = state['sigma']

                # Update eigenvector estimate via power iteration
                r = torch.sum(grad * v)  # Rayleigh quotient
                v_update = r * grad
                v_norm = v_update.norm() + epsilon
                v_hat = v_update / v_norm

                # Moving average update for eigenvector
                v.mul_(beta1).add_(v_hat, alpha=1 - beta1)
                v_norm = v.norm() + epsilon
                v.div_(v_norm)  # Maintain unit norm

                # Update eigenvalue estimate (exponential moving average)
                sigma_update = r ** 2
                sigma.mul_(beta2).add_(sigma_update, alpha=1 - beta2)
                sigma_hat = sigma / (1 - beta2 ** state['step'])  # Bias correction

                # Compute spectral preconditioning factor
                preconditioner = 1.0 / (torch.sqrt(sigma_hat) + epsilon)

                # Apply parameter update
                p.add_(grad * (-lr * preconditioner))

        return loss