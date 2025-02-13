# pylint:disable=signature-differs, not-callable # type:ignore
import itertools

import torch
from torch.optim import Optimizer

class RicciFlow(Optimizer):
    """hvp verson"""
    def __init__(self, params, lr=1e-3, curvature_rate=0.1, hvp_samples=2,
                 damping=1e-2, grad_clip=5.0):
        defaults = dict(lr=lr, curvature_rate=curvature_rate,
                        hvp_samples=hvp_samples, damping=damping,
                        grad_clip=grad_clip)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['metric'] = torch.ones_like(p.data)
                state['avg_curvature'] = torch.zeros_like(p.data)
    @torch.no_grad
    def step(self, closure):
        with torch.enable_grad():
            loss = closure(False)

            # First backward pass (retain graph for HvP)
            loss.backward(retain_graph=True, create_graph=True)

            params = []
            grads = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        params.append(p)
                        grads.append(p.grad.clone())

            # Compute Hessian diagonal via Hutchinson estimator
            hessian_diags = [torch.zeros_like(p) for p in params]

            for _ in range(self.defaults['hvp_samples']):
                # Generate Rademacher vectors (-1/+1)
                v = [torch.randint_like(p, high=2) * 2 - 1 for p in params]

                # Compute HvP = ∇(g·v)
                gv = sum([(g * v_p).sum() for g, v_p in zip(grads, v)])
                Hv = torch.autograd.grad(gv, params, retain_graph=True)

                # Accumulate diagonal estimate: E[v⊙Hv] = diag(H)
                for i, (v_p, Hv_p) in enumerate(zip(v, Hv)):
                    hessian_diags[i] += (v_p * Hv_p).detach()

        # Average across samples
        for hd in hessian_diags:
            hd.div_(self.defaults['hvp_samples'])

        # Parameter updates
        for group in self.param_groups:
            for p in params:
                if p.grad is None:
                    continue

                state = self.state[p]
                metric = state['metric']
                avg_curv = state['avg_curvature']
                hessian_diag = hessian_diags[params.index(p)]

                # Curvature normalization
                avg_curv.mul_(0.9).add_(hessian_diag.abs(), alpha=0.1)
                normalized_hessian = hessian_diag / (avg_curv + 1e-16)

                # Damped metric update
                metric_update = metric * (1 - group['curvature_rate'] * normalized_hessian)
                metric_update.add_(group['damping'] * (1 - metric))
                metric_update.clamp_(0.1, 10.0)
                state['metric'] = metric_update

                # Preconditioned gradient step
                g = torch.clamp(p.grad, -group['grad_clip'], group['grad_clip'])
                p.data.addcdiv_(-group['lr'], g, metric_update + 1e-8)

        # Cleanup graph to prevent memory leaks
        for p in params:
            p.grad = None
        torch.cuda.empty_cache()

        return loss