import torch

class InterstepSecant(torch.optim.Optimizer):
    """todo test high epsilon like 1"""
    def __init__(self, params, lr=1e-3, epsilon=1e-8):
        defaults = dict(lr=lr, epsilon=epsilon)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            epsilon = group['epsilon']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'prev_grad' not in state:
                    # First step: perform SGD and store the gradient
                    state['prev_grad'] = grad.clone().detach()
                    p.data.add_(-lr * grad)
                else:
                    prev_grad = state['prev_grad']
                    # Compute denominator with numerical stability
                    denom = grad - prev_grad
                    denom_abs = denom.abs()
                    # Ensure denominator's absolute value is at least epsilon
                    denom = torch.where(denom_abs < epsilon, torch.sign(denom) * epsilon, denom)
                    # Calculate the step and apply learning rate
                    step = (grad * prev_grad) / denom
                    step.mul_(lr)
                    # Update parameters
                    p.data.add_(step)
                    # Store current gradient for next step
                    state['prev_grad'] = grad.clone().detach()

        return loss