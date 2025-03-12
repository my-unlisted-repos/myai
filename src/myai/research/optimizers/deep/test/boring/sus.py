import torch
from torch.optim import Optimizer

class SUS(Optimizer):
    """
    Stochastic Update Significance. Maintains running estimates of gradient means and variances.
    Uses statistical significance testing (z-scores) to decide between current gradient or mean gradient for updates
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), threshold=1.0, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, threshold=threshold, eps=eps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            threshold = group['threshold']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SUS does not support sparse gradients')

                state = self.state[p]
                m, v = state['m'], state['v']
                step = state['step'] + 1

                # Update first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)

                # Compute variance and ensure non-negativity
                var = v_hat - m_hat.pow(2)
                var = var.clamp(min=0) + eps
                std = torch.sqrt(var)

                # Calculate z-score of current gradient
                z = (grad - m_hat) / (std + eps)

                # Significance-based update decision
                significant = torch.abs(z) > threshold
                update = torch.where(significant, grad, m_hat)

                # Apply adaptive learning rate
                if state['step'] > 1: step_size = lr / (std + eps)
                else: step_size = 1 / max(1, torch.linalg.norm(grad)) # pylint:disable=not-callable

                # Update parameters
                p.data.add_(-step_size * update)

                state['step'] = step

        return loss