import torch
from torch.optim import Optimizer


class GeoMean(Optimizer):
    """
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (default: 1e-3)
    """

    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_data = p.data

                # Component-wise geometric scaling of gradient
                scaled_grad = d_p * torch.sqrt(torch.abs(param_data) * torch.abs(d_p))

                # Parameter update
                p.data.add_( -group['lr'], scaled_grad)

        return loss