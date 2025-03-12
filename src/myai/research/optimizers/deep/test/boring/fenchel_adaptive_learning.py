import torch

class FenchelAdaptiveLearning(torch.optim.Optimizer):
    """Implements the Fenchel Adaptive Learning (FAL) optimizer.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): exponential decay rate for the first moment estimates (default: 0.9)
        beta2 (float, optional): exponential decay rate for the second moment estimates (default: 0.999)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['y'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)  # First moment (accumulated gradients)
                state['a'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)  # Second moment (curvature estimate)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                # Update first moment (y)
                state['y'] = beta1 * state['y'] + (1 - beta1) * grad

                # Update second moment (a)
                state['a'] = beta2 * state['a'] + (1 - beta2) * grad.pow(2)

                # Bias correction (optional, can be removed for non-bias-corrected version)
                y_hat = state['y'] / (1 - beta1 ** state['step'])
                a_hat = state['a'] / (1 - beta2 ** state['step'])

                # Compute the update using the Fenchel conjugate scaling
                denom = a_hat + eps
                step_size = lr * ((1 - beta2 ** state['step']) ** 0.5) / (1 - beta1 ** state['step'])  # Optional bias correction
                p.data.addcdiv_(-step_size, y_hat, denom)

        return loss