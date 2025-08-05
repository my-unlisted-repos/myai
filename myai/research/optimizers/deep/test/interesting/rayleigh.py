import torch
from torch.optim import Optimizer

class RayleighIteration(Optimizer):
    """Implements the Rayleigh Iteration Optimizer (RIO) algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta2 (float, optional): EMA coefficient for squared gradients (default: 0.999)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, beta2=0.999, eps=1e-8):

        defaults = dict(lr=lr, beta2=beta2, eps=eps)
        super().__init__(params, defaults)

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
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RayleighIteration does not support sparse gradients')

                state = self.state[p]

                # Initialize state if necessary
                if len(state) == 0:
                    state['step'] = 0
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['rho_prev'] = torch.tensor(0.0, device=p.device)

                v = state['v']
                rho_prev = state['rho_prev']
                beta2 = group['beta2']
                eps = group['eps']
                lr = group['lr']

                state['step'] += 1

                # Update the EMA of squared gradients
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute current Rayleigh quotient (rho)
                g_squared = grad ** 2
                numerator = torch.sum(g_squared * v)
                denominator_rho = torch.sum(g_squared)

                if denominator_rho == 0:
                    rho = torch.tensor(0.0, device=p.device)
                else:
                    rho = numerator / denominator_rho

                # Compute denominator with absolute value for stability
                denom_update = (v - rho_prev).abs().add(eps)

                # Update parameters
                p.add_(grad / denom_update, alpha=-lr)

                # Update stored rho_prev for next iteration
                state['rho_prev'].copy_(rho.detach())

        return loss