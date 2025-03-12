import torch
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector

def vector_to_parameters(vec, parameters):
    offset = 0
    for p in parameters:
        numel = p.numel()
        p.data.copy_(vec[offset:offset+numel].view_as(p))
        offset += numel
    return parameters

class StableLBFGS(Optimizer):
    def __init__(self, params, lr=1.0, history_size=10, norm_growth_bound=2., pos = 1.1, neg = 0.9):
        defaults = dict(lr=lr, history_size=history_size, norm_growth_bound=norm_growth_bound)
        super().__init__(params, defaults)

        self.state['global'] = {
            'history': [],
            'prev_dir': None,
            'prev_norm': 1e-3,
            'step': 0
        }

        self.adapt_value = 1
        self.pos = pos
        self.neg = neg

    @torch.no_grad
    def step(self, closure):
        if closure is None:
            raise ValueError("Closure required for StableLBFGS")

        # First closure call to get initial loss and gradient
        with torch.enable_grad():
            loss = closure()

        params = [p for group in self.param_groups for p in group['params']]
        g1 = [p.grad.detach().clone() for p in params]

        state = self.state['global']
        history = state['history']
        prev_dir = state['prev_dir']
        lr = self.defaults['lr'] * self.adapt_value
        norm_growth = self.defaults['norm_growth_bound']

        # Determine trial step direction
        if prev_dir is None:
            # First step: use scaled negative gradient
            trial_step = [lr * g * 1e-3 for g in g1 ]
        else:
            trial_step = [d * lr for d in prev_dir]

        # Apply trial step temporarily
        self._apply_step(trial_step, 1.0)

        # Second closure call to get gradient after trial step
        with torch.enable_grad():
            loss2 = closure()

        if prev_dir is not None:
            if loss2 < loss: self.adapt_value *= self.pos
            else: self.adapt_value *= self.neg

        g2 = [p.grad.detach().clone() for p in params]

        # Revert trial step
        self._apply_step(trial_step, -1.0)

        # Compute curvature pair
        y = [g2_p - g1_p for g2_p, g1_p in zip(g2, g1)]
        s = trial_step

        # Maintain history size
        history.append((s, y))
        if len(history) > self.defaults['history_size']:
            history.pop(0)

        # Restore original gradients
        for p, g in zip(params, g1):
            p.grad = g.clone()

        # Compute L-BFGS direction
        dir = self._compute_direction(g1)

        # Clip direction norm
        dir_norm = torch.norm(torch.stack([t.norm() for t in dir]))
        max_norm = state['prev_norm'] * norm_growth if state['prev_norm'] else dir_norm

        if state['prev_norm'] and dir_norm > max_norm:
            scale = max_norm / dir_norm
            dir = [d * scale for d in dir]
            dir_norm = max_norm

        # Update state
        state['prev_dir'] = [d.detach().clone() for d in dir]
        state['prev_norm'] = dir_norm
        state['step'] += 1

        # Apply final update
        self._apply_step(dir, lr)

        return loss

    def _apply_step(self, direction, scale):
        for p, d in zip(self._params, direction):
            p.data.add_(d, alpha=scale)

    def _compute_direction(self, grad):
        history = self.state['global']['history']
        if not history:
            return [-g for g in grad]

        # Flatten tensors for vector operations
        flat_grad = parameters_to_vector(grad)
        q = flat_grad.clone()
        alpha_list = []

        # First loop (reversed history)
        for s, y in reversed(history):
            flat_s = parameters_to_vector(s)
            flat_y = parameters_to_vector(y)
            rho = 1.0 / flat_y.dot(flat_s)
            alpha = rho * flat_s.dot(q)
            q.add_(flat_y, alpha=-alpha)
            alpha_list.append(alpha)

        # Initial Hessian approximation
        s_last, y_last = history[-1]
        flat_s = parameters_to_vector(s_last)
        flat_y = parameters_to_vector(y_last)
        gamma = flat_s.dot(flat_y) / flat_y.dot(flat_y)
        r = q * gamma

        # Second loop
        for (s, y), alpha in zip(history, reversed(alpha_list)):
            flat_s = parameters_to_vector(s)
            flat_y = parameters_to_vector(y)
            rho = 1.0 / flat_y.dot(flat_s)
            beta = rho * flat_y.dot(r)
            r.add_(flat_s, alpha=alpha - beta)

        # Convert back to parameter list
        direction = [-r]  # Negative for descent
        direction = vector_to_parameters(-r, grad)
        return list(direction)

    @property
    def _params(self):
        return [p for group in self.param_groups for p in group['params']]