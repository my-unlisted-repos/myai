import torch


class MirroredSGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure): # pylint:disable = signature-differs
        with torch.enable_grad():
            loss = closure()

        for g in self.param_groups:
            for p in g['params']:
                p.neg_()
                if p.grad is not None:
                    state = self.state[p]
                    state['grad'] = p.grad

        with torch.enable_grad():
            loss = closure()


        for g in self.param_groups:
            for p in g['params']:
                p.neg_()
                if p.grad is not None:
                    state = self.state[p]
                    update = state['grad'] - p.grad

                    p.sub_(update, alpha = g['lr'])

        return loss

class AlternatingMirroredSGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for g in self.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    p.sub_(p.grad, alpha = g['lr'])
                p.neg_()

        return loss

