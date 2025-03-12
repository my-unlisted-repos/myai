import torch
from torch.optim import Optimizer

class EMA_LBFGS(Optimizer):
    def __init__(self, params, lr=1.0, beta_params=0.9, beta_grads=0.9, memory_size=10, damping=1e-8, max_iter=20, tolerance_grad=1e-9, tolerance_change=1e-9, line_search_fn=None, norm_bound:float|None = 1, value_bound:float|None = 1, ):
        defaults = dict(
            lr=lr,
            beta_params=beta_params,
            beta_grads=beta_grads,
            memory_size=memory_size,
            damping=damping,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
            value_bound = value_bound,
            norm_bound = norm_bound,
        )

        super(EMA_LBFGS, self).__init__(params, defaults)
        self.state['global_step'] = 0
        self.history = []

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['ema_param'] = torch.zeros_like(p.data)
                state['ema_grad'] = torch.zeros_like(p.data)
                state['prev_debiased_param'] = torch.zeros_like(p.data)
                state['prev_debiased_grad'] = torch.zeros_like(p.data)

    def _get_debiased_ema(self, ema, beta, step):
        return ema / (1 - beta ** step) if step > 0 else ema

    def step(self, closure):
        global_step = self.state['global_step']
        self.state['global_step'] += 1

        # Compute loss and gradients
        loss = closure()
        loss = float(loss)

        # Update EMA gradients and compute debiased
        for group in self.param_groups:
            beta_grads = group['beta_grads']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['ema_grad'] = state['ema_grad'] * beta_grads + (1 - beta_grads) * p.grad.data
                debiased_grad = self._get_debiased_ema(state['ema_grad'], beta_grads, global_step)
                state['y'] = debiased_grad - state['prev_debiased_grad']
                state['prev_debiased_grad'] = debiased_grad.clone()

        # Collect current gradients (debiased EMAs)
        current_grad = []
        for group in self.param_groups:
            for p in group['params']:
                current_grad.append(self.state[p]['prev_debiased_grad'].clone())

        # Two-loop recursion to compute search direction
        q = [g.neg() for g in current_grad]
        history = self.history
        if len(history) > 0:
            alphas = []
            for s, y in reversed(history):
                rho = 1.0 / self._dot(y, s)
                alpha = rho * self._dot(s, q)
                alphas.append(alpha)
                q = self._add(q, self._mul(y, -alpha))

            # Scale initial Hessian approximation
            if len(history) > 0:
                s, y = history[-1]
                ys = self._dot(y, s)
                scale = ys / self._dot(y, y)
                q = self._mul(q, scale)

            for s, y in history:
                rho = 1.0 / self._dot(y, s)
                beta = rho * self._dot(y, q)
                q = self._add(q, self._mul(s, alphas.pop() - beta))

            # clipping
            value_bound = group['value_bound']
            norm_bound = group['norm_bound']

            if norm_bound is not None:
                for i in q:
                    norm = torch.linalg.vector_norm(i)
                    if norm > norm_bound:
                        i.div_(norm/norm_bound)
            if value_bound is not None:
                for i in q: i.clip_(-value_bound, value_bound)


        else: # dampen 1st GD step
            q = [i*1e-3 for i in q]
        direction = q

        # Line search
        lr = group['lr']
        params = [p.data for p in group['params']]
        orig_loss = loss
        flat_direction = self._flatten(direction)

        with torch.no_grad():
            if group['line_search_fn'] is not None:
                # Perform line search
                pass  # Implement line search similar to LBFGS
            else:
                # No line search, take fixed step
                self._add_update(flat_direction, lr, params)

        # Update EMA parameters
        for group in self.param_groups:
            beta_params = group['beta_params']
            for p in group['params']:
                state = self.state[p]
                state['ema_param'] = state['ema_param'] * beta_params + (1 - beta_params) * p.data
                debiased_param = self._get_debiased_ema(state['ema_param'], beta_params, global_step)
                state['s'] = debiased_param - state['prev_debiased_param']
                state['prev_debiased_param'] = debiased_param.clone()

        # Compute curvature and update history
        s_list = [self.state[p]['s'] for p in group['params']]
        y_list = [self.state[p]['y'] for p in group['params']]
        curvature = self._dot(s_list, y_list)

        if curvature < group['damping']:
            # Apply damping
            y_list = [y + group['damping'] * s for y, s in zip(y_list, s_list)]
            curvature = self._dot(s_list, y_list)

        if curvature > 1e-10:
            self.history.append((s_list, y_list))
            if len(self.history) > group['memory_size']:
                self.history.pop(0)

        return loss

    def _dot(self, x, y):
        return sum(torch.dot(a.flatten(), b.flatten()) for a, b in zip(x, y))

    def _add(self, x, y, alpha=1.0):
        return [a + alpha * b for a, b in zip(x, y)]

    def _mul(self, x, alpha):
        return [a * alpha for a in x]

    def _flatten(self, x):
        return torch.cat([a.flatten() for a in x])

    def _add_update(self, direction, lr, params):
        offset = 0
        for p in params:
            numel = p.numel()
            p.add_(direction[offset:offset+numel].view_as(p), alpha=lr)
            offset += numel