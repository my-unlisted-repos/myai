# pylint:disable=signature-differs, not-callable # type:ignore
"""there are 2 versions!"""

import torch
from torch.optim import Optimizer

class StochasticLBFGS(Optimizer):
    """This might work on problems with a bit of noise, but not mini-batch levels of noise.
    And this doesn't use closure. Better than LBFGS and SGD on some tasks."""
    def __init__(self, params, lr=0.1, warmup_steps=5, buffer_size=5,
                 min_damping=1e-3, max_lr_scale=10.0):
        defaults = dict(lr=lr, warmup_steps=warmup_steps, buffer_size=buffer_size,
                        min_damping=min_damping, max_lr_scale=max_lr_scale)
        super().__init__(params, defaults)

        self.state['step'] = 0
        self.state['buffer'] = []
        self.state['prev_params'] = None
        self.state['prev_grads'] = None

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Retrieve current parameters and gradients
        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                grads.append(p.grad.detach().clone())

        # Initialize previous state
        if self.state['prev_params'] is None:
            self.state['prev_params'] = [p.detach().clone() for p in params]
            self.state['prev_grads'] = [g.detach().clone() for g in grads]
            self.state['step'] += 1
            return loss

        # Compute s and y with 1-step delayed gradients
        s = [p - p_prev for p, p_prev in zip(params, self.state['prev_params'])]
        y = [g - g_prev for g, g_prev in zip(grads, self.state['prev_grads'])]

        # Apply adaptive damping
        sy = self._vdot(s, y)
        ss = self._vdot(s, s)
        damping = max(self.defaults['min_damping'], abs(sy)/(ss + 1e-8))
        y = [y_i + damping * s_i for y_i, s_i in zip(y, s)]

        # Store delayed (s, y) pair
        self.state['buffer'].append((s, y))
        if len(self.state['buffer']) > self.defaults['buffer_size']:
            self.state['buffer'].pop(0)

        # Save current state for next iteration (before parameter update)
        self.state['prev_params'] = [p.detach().clone() for p in params]
        self.state['prev_grads'] = [g.detach().clone() for g in grads]

        # Warmup phase: Use SGD with decaying learning rate
        if self.state['step'] < self.defaults['warmup_steps']:
            lr = self.defaults['lr'] * (self.state['step'] / self.defaults['warmup_steps'])
            self._sgd_update(params, grads, lr)
            self.state['step'] += 1
            return loss

        # L-BFGS phase
        d = self._two_loop_recursion(grads)
        self._adaptive_update(params, d, grads)

        self.state['step'] += 1
        return loss

    def _two_loop_recursion(self, grads):
        d = [-g.clone() for g in grads]
        alphas = []
        buffer = self.state['buffer']

        # First loop (reverse order)
        for i in reversed(range(len(buffer))):
            s_i, y_i = buffer[i]
            sy = self._vdot(s_i, y_i)
            rho = 1.0 / (sy + 1e-8)
            alpha = rho * self._vdot(s_i, d)
            alpha = torch.clamp(alpha, -1e3, 1e3)  # Clip extreme values
            alphas.append(alpha)
            d = [dj - alpha * yij for dj, yij in zip(d, y_i)]

        # Scale initial direction
        if len(buffer) > 0:
            s_last, y_last = buffer[-1]
            sy = self._vdot(s_last, y_last)
            yy = self._vdot(y_last, y_last)
            gamma = sy / (yy + 1e-8)
            gamma = torch.clamp(gamma, 1e-6, 1e6)  # Prevent division by near-zero
            d = [gamma * di for di in d]

        # Second loop (original order)
        for i in range(len(buffer)):
            s_i, y_i = buffer[i]
            sy = self._vdot(s_i, y_i)
            rho = 1.0 / (sy + 1e-8)
            beta = rho * self._vdot(y_i, d)
            alpha = alphas.pop()
            d = [dj + (alpha - beta) * sij for dj, sij in zip(d, s_i)]

        return d

    def _adaptive_update(self, params, direction, grads):
        # Auto-scale learning rate based on gradient magnitude
        grad_norm = sum(g.norm()**2 for g in grads).sqrt()
        dir_norm = sum(d.norm()**2 for d in direction).sqrt()
        lr_scale = torch.clamp(grad_norm / (dir_norm + 1e-8),
                              max=self.defaults['max_lr_scale'])
        lr = self.defaults['lr'] * lr_scale.item()

        # Apply update with NaN check
        for p, d in zip(params, direction):
            if torch.isnan(d).any():
                d = -p.grad  # Fallback to SGD on NaN
            p.add_(d, alpha=lr)

    def _sgd_update(self, params, grads, lr):
        for p, g in zip(params, grads):
            p.data.add_(g, alpha=-lr)

    def _vdot(self, a, b):
        return sum(torch.dot(ai.flatten(), bi.flatten()) for ai, bi in zip(a, b))


class SingleStepLBFGS(Optimizer):
    """LBFGS version that doesn't use closure. Better than LBFGS and SGD on some tasks.
    If LBFGS goes unstable with lr = 1, this one might still work and beat it."""
    def __init__(self, params, lr=1., history_size=5, epsilon=1e-8):
        defaults = dict(lr=lr, history_size=history_size, epsilon=epsilon)
        super().__init__(params, defaults)
        self.state['step'] = 0
        self.state['s_history'] = []
        self.state['y_history'] = []
        self.state['rho_history'] = []
        self.state['prev_flat_grad'] = None
        self.state['prev_flat_params'] = None

    def _gather_flat_grad(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                views.append(p.grad.data.view(-1))
        return torch.cat(views) if views else torch.tensor([])

    def _gather_flat_params(self):
        views = []
        for group in self.param_groups:
            for p in group['params']:
                views.append(p.data.view(-1))
        return torch.cat(views) if views else torch.tensor([])

    def _distribute_flat_params(self, flat_params):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.data.numel()
                p.data.copy_(flat_params[offset:offset+numel].view_as(p.data))
                offset += numel

    @torch.no_grad
    def step(self, closure=None):
        # closure = torch.enable_grad()(closure)
        self.state['step'] += 1

        # Compute current gradient
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        # self.zero_grad()
        # loss.backward()
        flat_grad = self._gather_flat_grad()
        flat_params = self._gather_flat_params()

        # Initialize state for first step
        if self.state['step'] == 1:
            self.state['prev_flat_grad'] = flat_grad.detach().clone()
            self.state['prev_flat_params'] = flat_params.detach().clone()
            return loss

        # Update history with s and y
        s = flat_params - self.state['prev_flat_params']
        y = flat_grad - self.state['prev_flat_grad']

        history_size = self.defaults['history_size']
        if len(self.state['s_history']) >= history_size:
            self.state['s_history'].pop(0)
            self.state['y_history'].pop(0)
            self.state['rho_history'].pop(0)

        ys = torch.dot(y, s)
        if ys > self.defaults['epsilon']:
            self.state['s_history'].append(s)
            self.state['y_history'].append(y)
            self.state['rho_history'].append(1.0 / ys)

        # Compute search direction via L-BFGS two-loop recursion
        q = flat_grad.neg()
        alpha = []
        for s, y, rho in zip(reversed(self.state['s_history']),
                            reversed(self.state['y_history']),
                            reversed(self.state['rho_history'])):
            alpha_i = rho * torch.dot(s, q)
            q.add_(y, alpha=-alpha_i)
            alpha.append(alpha_i)

        if self.state['s_history']:
            gamma = self.state['s_history'][-1].dot(self.state['y_history'][-1]) / self.state['y_history'][-1].dot(self.state['y_history'][-1])
            z = q * gamma
        else:
            z = q

        for s, y, rho, alpha_i in zip(self.state['s_history'],
                                    self.state['y_history'],
                                    self.state['rho_history'],
                                    reversed(alpha)):
            beta = rho * torch.dot(y, z)
            z.add_(s, alpha=alpha_i - beta)

        # Update parameters
        self._distribute_flat_params(flat_params + self.defaults['lr'] * z)

        # Update previous state
        self.state['prev_flat_grad'] = flat_grad.detach().clone()
        self.state['prev_flat_params'] = flat_params.detach().clone()

        return loss