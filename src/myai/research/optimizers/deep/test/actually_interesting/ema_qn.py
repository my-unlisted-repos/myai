import torch
from torch.optim import Optimizer

class EMAQN(Optimizer):
    def __init__(self, params, lr=0.1, beta=0.9, history_size=10,
                 damping=1e-4, use_sr1=False, norm_growth_bound: float | None = 2, ord = 1):
        defaults = dict(lr=lr, beta=beta, history_size=history_size,
                        damping=damping, use_sr1=use_sr1)
        super().__init__(params, defaults)

        self._params = [p for group in self.param_groups for p in group['params']]
        self.state = self.state['global_state']
        self.state.setdefault('step', 0)
        self.state.setdefault('ema_grad', None)
        self.state.setdefault('prev_params', None)
        self.state.setdefault('prev_true_grad', None)
        self.state.setdefault('history', [])
        self.norm_growth_bound = norm_growth_bound
        self.ord = ord
        self.prev_norm = None

    def _gather_flat(self, prop):
        return torch.cat([getattr(p, prop).data.view(-1) for p in self._params])

    def _distribute(self, flat_tensor, prop):
        offset = 0
        for p in self._params:
            numel = p.numel()
            getattr(p, prop).data.copy_(flat_tensor[offset:offset+numel].view_as(getattr(p, prop)))
            offset += numel

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        state = self.state
        group = self.param_groups[0]
        beta, damping = group['beta'], group['damping']

        current_params = self._gather_flat('data')
        current_grad = self._gather_flat('grad')
        t = state['step'] + 1

        # EMA Gradient with Bias Correction
        if state['ema_grad'] is None:
            state['ema_grad'] = current_grad.clone()
        else:
            state['ema_grad'].mul_(beta).add_(current_grad, alpha=1 - beta)

        ema_grad = state['ema_grad'] / (1 - beta**t)  # Bias correction

        if state['prev_params'] is not None:
            # Compute curvature pair with true gradients
            s = current_params - state['prev_params']
            y = current_grad - state['prev_true_grad']

            # Apply Powell damping
            curv = y.dot(s)
            if curv < (0.2 if group['use_sr1'] else 0.001) * s.dot(s):
                y.add_(s, alpha=damping)
                curv = y.dot(s)

            if curv > 1e-12:
                state['history'].append((s, y))
                if len(state['history']) > group['history_size']:
                    state['history'].pop(0)

        # Compute search direction
        if not state['history']:
            d = ema_grad
        else:
            if group['use_sr1']:
                d = self._sr1_direction(ema_grad)
            else:
                d = self._lbfgs_direction(ema_grad)

        # first step is a tiny GD step
        if t == 1: new_params = current_params - (group['lr'] * d) * 1e-4

        # Update parameters
        else:
            if self.norm_growth_bound is not None:
                # bound norm growth
                norm = torch.linalg.vector_norm(d, ord = self.ord)
                if self.prev_norm is None:
                    self.prev_norm = norm

                else:
                    growth = norm / self.prev_norm
                    if growth > self.norm_growth_bound:
                        d /= growth / self.norm_growth_bound
                        self.prev_norm = self.prev_norm * self.norm_growth_bound
                    else:
                        self.prev_norm = norm

            new_params = current_params + group['lr'] * d

        self._distribute(new_params, 'data')

        # Store previous values
        state['prev_params'] = current_params.clone()
        state['prev_true_grad'] = current_grad.clone()
        state['step'] = t

        return loss

    def _lbfgs_direction(self, grad):
        history = self.state['history']
        q = grad.clone()
        alpha = []

        # Reverse recursion (newest to oldest)
        for s, y in reversed(history):
            rho = 1.0 / y.dot(s)
            alpha_i = rho * s.dot(q)
            q.add_(y, alpha=-alpha_i)
            alpha.append(alpha_i)

        # Reverse alpha to match history order
        alpha = alpha[::-1]

        # Initial scaling
        if history:
            s, y = history[-1]
            gamma = y.dot(s) / y.dot(y)
            r = gamma * q
        else:
            r = q

        # Forward recursion (oldest to newest)
        for i, (s, y) in enumerate(history):
            rho = 1.0 / y.dot(s)
            beta_i = rho * y.dot(r)
            r.add_(s, alpha=alpha[i] - beta_i)

        return -r

    def _sr1_direction(self, grad):
        d = grad.clone()
        for s, y in reversed(self.state['history']):
            Bs = self._sr1_operator(s)
            curv = s.dot(Bs - s)
            if abs(curv) > 1e-8:
                d.add_((Bs - s) * (s.dot(d)) / curv)
        return -d

    def _sr1_operator(self, v):
        result = v.clone()
        for s, y in self.state['history']:
            ys = y - s
            result.add_(ys * ys.dot(v) / ys.dot(s))
        return result