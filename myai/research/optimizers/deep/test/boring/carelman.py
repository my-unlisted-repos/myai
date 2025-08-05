import torch
from torch.optim import Optimizer

class CarlemanOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, gamma=0.1, epsilon=1e-8):
        defaults = dict(lr=lr, gamma=gamma, epsilon=epsilon)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['m'] = torch.zeros_like(p.data)
                self.state[p]['v'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            epsilon = group['epsilon']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                m, v = state['m'], state['v']

                # Compute squared norm of the gradient for this parameter
                grad_norm_sq = torch.sum(grad ** 2)

                # Compute adaptive decay factor lambda_t
                lambda_t = torch.exp(-gamma * grad_norm_sq)

                # Update first and second moment estimates
                m.mul_(lambda_t).add_(grad, alpha=1 - lambda_t)
                v.mul_(lambda_t).addcmul_(grad, grad, value=1 - lambda_t)

                # Compute the denominator for the update
                denom = torch.sqrt(v) + epsilon

                # Update parameters
                p.data.addcdiv_(m, denom, value=-lr)
        return loss

