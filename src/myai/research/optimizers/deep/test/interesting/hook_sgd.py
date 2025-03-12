import torch
from torch.optim import Optimizer

class HookSGD(Optimizer):
    """
    this updates parameters DURING gradient computation with a hook. Will that affect backprop?
    """

    def __init__(self, params, lr=1e-2, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

        # Register backward hooks for each parameter
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['momentum_buffer'] = torch.zeros_like(p)
                    # Register hook to trigger on gradient computation
                    p.register_hook(self._make_param_hook(p, group))

    def _make_param_hook(self, p, group):
        @torch.no_grad
        def hook(grad):
            state = self.state[p]
            lr = group['lr']
            momentum = group['momentum']

            # Update momentum buffer
            state['momentum_buffer'].mul_(momentum).add_(grad)

            # Update parameter in-place
            p.sub_(lr * state['momentum_buffer'])

            return grad  # Return gradient unmodified
        return hook

    def step(self, closure):
        """No-op. Updates are performed during backward pass via hooks."""
        return closure()