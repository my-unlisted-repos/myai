import torch
from torch.optim import Optimizer

class LowRankAdam(Optimizer):
    """
    approximates the preconditioning matrix as a diagonal plus a rank-1 matrix, updated via
    exponential moving averages using the Sherman-Morrison formula

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        beta1 (float, optional): Exponential decay rate for the diagonal term (default: 0.9).
        beta2 (float, optional): Exponential decay rate for the low-rank term (default: 0.999).
        eps (float, optional): Term added to denominators to prevent division by zero (default: 1e-8).
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('LowRankAdam does not support sparse gradients')

                state = self.state[p]

                # Initialize state if required
                if len(state) == 0:
                    state['step'] = 0
                    state['v'] = torch.zeros_like(p)
                    state['u'] = torch.zeros_like(p)

                v, u = state['v'], state['u']
                beta1, beta2 = group['beta1'], group['beta2']
                eps = group['eps']
                lr = group['lr']
                state['step'] += 1
                t = state['step']

                # Update diagonal term v (similar to RMSProp)
                v.mul_(beta1).addcmul_(grad, grad, value=1 - beta1)

                # Update low-rank term u (exponential moving average of gradients)
                u.mul_(beta2).add_(grad, alpha=1 - beta2)

                # Bias correction
                v_corrected = v / (1 - beta1 ** t)
                u_corrected = u / (1 - beta2 ** t)

                # Compute inverse of diagonal term
                inv_v = 1.0 / (v_corrected + eps)

                # Compute scaled low-rank vector
                u_scaled = inv_v * u_corrected

                # Numerator: (grad * inv_v) · u_corrected
                numerator = (grad * inv_v).dot(u_corrected)

                # Denominator: 1 + u_corrected · (inv_v * u_corrected)
                denominator = 1.0 + u_corrected.dot(u_scaled)

                # Sherman-Morrison update
                term1 = inv_v * grad
                term2 = (u_scaled * numerator) / denominator
                preconditioned_grad = term1 - term2

                # Apply update
                p.add_(preconditioned_grad, alpha=-lr)

        return loss