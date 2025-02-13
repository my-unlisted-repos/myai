# pylint:disable=signature-differs, not-callable # type:ignore

import torch
from torch.optim import Optimizer

def conjugate_gradient(Hvp_fn, b, max_iter=10, tol=1e-5):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.dot(r, r)
    for i in range(max_iter):
        Ap = Hvp_fn(p)
        alpha = rsold / (torch.dot(p, Ap) + 1e-10)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

class NewtonCG(Optimizer):
    def __init__(self, params, lr:float=1, cg_max_iter=10, cg_tol=1e-5):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, cg_max_iter=cg_max_iter, cg_tol=cg_tol)
        super().__init__(params, defaults)
        self._params = [p for group in self.param_groups for p in group['params']]

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = torch.zeros_like(p.data).view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _update_params(self, delta):
        idx = 0
        for p in self._params:
            plen = p.data.numel()
            p_update = delta[idx: idx + plen].view_as(p.data)
            p.data.add_(p_update, alpha=self.param_groups[0]['lr'])
            idx += plen

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for EquilibriumOptimizer")

        # Compute loss and gradients with create_graph=True
        loss = closure(False)
        self.zero_grad()
        loss.backward(create_graph=True)

        # Gather flat gradient
        g = self._gather_flat_grad()

        # Define Hessian-vector product function
        def Hvp_fn(v):
            self.zero_grad()
            gv = torch.dot(g, v)
            hv = torch.autograd.grad(gv, self._params, retain_graph=True, allow_unused=True)
            hv_flat = torch.cat([h.contiguous().view(-1) if h is not None else torch.zeros_like(p) for h, p in zip(hv, self._params)])
            return hv_flat

        # Run conjugate gradient to solve HΔθ = -g
        group = self.param_groups[0]
        delta = conjugate_gradient(
            Hvp_fn,
            -g,
            max_iter=group['cg_max_iter'],
            tol=group['cg_tol']
        )

        # Update parameters with the CG result scaled by learning rate
        self._update_params(delta)

        return loss