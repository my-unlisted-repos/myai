# pylint:disable=signature-differs, not-callable

from collections import defaultdict

import torch
from torch.optim import Optimizer


class Homeomorphic(Optimizer):
    """1"""
    def __init__(self, params, lr=1e-3, alpha=0.1, beta=1.0, num_iters=5):
        defaults = dict(lr=lr, alpha=alpha, beta=beta, num_iters=num_iters)
        super().__init__(params, defaults)
        self.state = defaultdict(dict)

        self.current_step = 0

    def _calc_phi(self):
        # First pass: compute phi from current theta and save in state
        for group in self.param_groups:
            alpha = group['alpha']
            beta = group['beta']
            for p in group['params']:
                if not p.requires_grad: continue
                state = self.state[p]
                # Compute phi = h(theta) = theta + alpha * tanh(theta / beta)
                theta = p.data
                phi = theta + alpha * torch.tanh(theta / beta)
                state['phi'] = phi.detach()

        # Set parameters to theta computed from phi
        for group in self.param_groups:
            alpha = group['alpha']
            beta = group['beta']
            num_iters = group.get('num_iters', 5)
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                phi = state['phi']
                # Compute theta from phi via Newton-Raphson
                theta = phi.clone()
                for _ in range(num_iters):
                    residual = theta + alpha * torch.tanh(theta / beta) - phi
                    derivative = 1 + (alpha / beta) * (1.0 / torch.cosh(theta / beta)) ** 2
                    theta = theta - residual / derivative
                p.data.copy_(theta)

    @torch.no_grad
    def step(self, closure=None):

        # Compute loss and gradients
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        if self.current_step == 0: self._calc_phi()

        # Update phi based on gradients
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                phi = state['phi']
                theta = p.data

                # Compute dtheta/dphi = 1 / (dphi/dtheta)
                sech_sq = (1.0 / torch.cosh(theta / beta)) ** 2
                dphi_dtheta = 1 + (alpha / beta) * sech_sq
                dtheta_dphi = 1.0 / dphi_dtheta

                # Compute gradient w.r.t. phi: dL/dphi = dL/dtheta * dtheta/dphi
                dL_dphi = p.grad.data * dtheta_dphi

                # Update phi: phi_new = phi - lr * dL_dphi
                phi_new = phi - lr * dL_dphi
                state['phi'] = phi_new.detach()

                # Compute new theta from phi_new for the next iteration
                num_iters = group['num_iters']
                theta_new = phi_new.clone()
                for _ in range(num_iters):
                    residual = theta_new + alpha * torch.tanh(theta_new / beta) - phi_new
                    derivative = 1 + (alpha / beta) * (1.0 / torch.cosh(theta_new / beta)) ** 2
                    theta_new = theta_new - residual / derivative
                p.data.copy_(theta_new)

        self._calc_phi()
        self.current_step += 1

        return loss