from collections import deque

import torch
from torch.optim import Optimizer


class StochasticQuasiNewton(Optimizer):
    def __init__(self, params, lr=1e-3, history_size=10, tentative_lr=0.1):
        defaults = dict(lr=lr, history_size=history_size, tentative_lr=tentative_lr)
        super().__init__(params, defaults)
        self.history = deque(maxlen=history_size)
        self.tentative_lr = tentative_lr

    @torch.no_grad
    def step(self, closure):
        if closure is None:
            raise ValueError("Closure required for QuasiNewton optimizer")

        # Compute loss and gradient at current parameters
        with torch.enable_grad(): loss = closure()
        # self.zero_grad()
        # loss.backward()
        current_params = self._gather_flat_params()
        current_grad = self._gather_flat_grad()

        # Take tentative step
        tentative_params = current_params - self.tentative_lr * current_grad
        self._set_flat_params(tentative_params)

        # Compute loss and gradient at tentative parameters
        with torch.enable_grad(): loss_tentative = closure()
        # self.zero_grad()
        # loss_tentative.backward()
        tentative_grad = self._gather_flat_grad()

        # Compute s and y
        s = tentative_params - current_params
        y = tentative_grad - current_grad

        # Store curvature pair
        self.history.append((s, y))

        # Restore original parameters
        self._set_flat_params(current_params)

        # Two-loop recursion to compute search direction
        q = current_grad.clone()
        alphas = []

        # First loop (reverse)
        for s_i, y_i in reversed(self.history):
            rho_i = 1.0 / torch.dot(y_i, s_i)
            alpha_i = rho_i * torch.dot(s_i, q)
            alphas.append(alpha_i)
            q.add_(y_i, alpha=-alpha_i)

        # Scale by gamma
        if self.history:
            s_last, y_last = self.history[-1]
            gamma = torch.dot(y_last, s_last) / torch.dot(y_last, y_last)
            q.mul_(gamma)
        else:
            q.zero_()

        # Second loop (forward)
        for idx, (s_i, y_i) in enumerate(self.history):
            rho_i = 1.0 / torch.dot(y_i, s_i)
            beta_i = rho_i * torch.dot(y_i, q)
            alpha = alphas[-(idx + 1)]
            q.add_(s_i, alpha=alpha - beta_i)

        direction = -q
        new_params = current_params + self.defaults['lr'] * direction
        self._set_flat_params(new_params)

        return loss

    def _gather_flat_params(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                if p.data is None:
                    view = p.data.new(p.data.numel()).zero_()
                else:
                    view = p.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_grad(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    view = p.data.new(p.data.numel()).zero_()
                else:
                    view = p.grad.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    def _set_flat_params(self, flat_params):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.data.numel()
                p.data = flat_params[offset:offset + numel].view_as(p.data)
                offset += numel



class StochasticQuasiNewton2(Optimizer):
    def __init__(self, params, lr=1e-3, history_size=5, tentative_step=1e-3):
        defaults = dict(lr=lr, history_size=history_size, tentative_step=tentative_step)
        super().__init__(params, defaults)
        self.state.setdefault('history', [])

    @torch.no_grad
    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for StochasticQuasiNewton")

        group = self.param_groups[0]
        params = group['params']

        # Initial evaluation to get loss and gradients
        with torch.enable_grad(): loss = closure()

        # Save current parameters and gradients
        old_params = [p.detach().clone() for p in params]
        old_grads = [p.grad.detach().clone() for p in params]

        # Take tentative step
        tentative_step = group['tentative_step']
        with torch.no_grad():
            for p in params:
                p -= tentative_step * p.grad

        # Re-evaluate on same batch
        with torch.enable_grad(): loss_new = closure()

        # Capture new parameters and gradients
        new_params = [p.detach().clone() for p in params]
        new_grads = [p.grad.detach().clone() for p in params]

        # Revert parameters to original state
        with torch.no_grad():
            for p, old_p in zip(params, old_params):
                p.copy_(old_p)

        # Compute s and y
        s = [np - op for np, op in zip(new_params, old_params)]
        y = [ng - og for ng, og in zip(new_grads, old_grads)]

        # Check curvature condition
        sy = sum(torch.sum(s_i * y_i) for s_i, y_i in zip(s, y)).item()
        if sy <= 1e-10:
            return loss

        # Update history
        self.state['history'].append((s, y))
        if len(self.state['history']) > group['history_size']:
            self.state['history'].pop(0)

        # Two-loop recursion to compute search direction
        q = [g.clone() for g in old_grads]
        alpha = []

        # First loop (reverse order)
        for s_i, y_i in reversed(self.state['history']):
            rho_i = 1.0 / sum(torch.sum(s_ik * y_ik) for s_ik, y_ik in zip(s_i, y_i))
            a_i = rho_i * sum(torch.sum(s_ik * q_k) for s_ik, q_k in zip(s_i, q))
            alpha.append(a_i)
            for q_k, y_ik in zip(q, y_i):
                q_k.sub_(a_i * y_ik)

        # Compute gamma for initial Hessian scaling
        if self.state['history']:
            s_last, y_last = self.state['history'][-1]
            sy = sum(torch.sum(s_j * y_j) for s_j, y_j in zip(s_last, y_last))
            yy = sum(torch.sum(y_j * y_j) for y_j in y_last)
            gamma = sy / yy
        else:
            gamma = 1.0

        r = [gamma * q_k for q_k in q]

        # Second loop (natural order)
        for s_i, y_i in self.state['history']:
            rho_i = 1.0 / sum(torch.sum(s_ik * y_ik) for s_ik, y_ik in zip(s_i, y_i))
            b_i = rho_i * sum(torch.sum(y_ik * r_k) for y_ik, r_k in zip(y_i, r))
            a_i = alpha.pop()
            for r_k, s_ik in zip(r, s_i):
                r_k.add_(s_ik, alpha=a_i - b_i)

        # Update parameters
        lr = group['lr']
        with torch.no_grad():
            for p, d in zip(params, r):
                p.add_(d, alpha=-lr)

        return loss