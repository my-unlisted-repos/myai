import torch
from torch.optim import Optimizer

class CubicRegularizedEMA(Optimizer):
    """
    approximates the cubic regularization term using a moving average of past step magnitudes

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): exponential decay rate for gradient squared (default: 0.9)
        beta2 (float, optional): exponential decay rate for step magnitudes (default: 0.999)
        M (float, optional): cubic regularization coefficient (default: 1e-3)
        eps (float, optional): term added to denominator for numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, M=1e-3, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, M=M, eps=eps)
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['h_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                state['v_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

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
            beta1 = group['beta1']
            beta2 = group['beta2']
            M = group['M']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('CubicCRN does not support sparse gradients')

                state = self.state[p]

                # Update gradient squared moving average
                h_avg = state['h_avg']
                h_avg.mul_(beta1).addcmul_(grad, grad, value=1 - beta1)

                # Compute denominator components
                denom = h_avg.sqrt().add_(state['v_avg'], alpha=M).add_(eps)

                # Calculate and apply update step
                step = -lr * grad / denom
                p.data.add_(step)

                # Update step magnitude moving average
                v_avg = state['v_avg']
                v_avg.mul_(beta2).add_(step.abs(), alpha=1 - beta2)

        return loss