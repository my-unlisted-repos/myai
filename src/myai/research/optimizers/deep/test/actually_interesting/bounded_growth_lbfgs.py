import torch
from torch.optim import Optimizer

class BoundedGrowthLBFGS(Optimizer):
    """
    Implements L-BFGS algorithm with element-wise bounded update growth.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (ignored for standard L-BFGS, but kept for API consistency)
        history_size (int): size of the history buffer
        norm_growth_bound (float): factor to bound the element-wise update magnitude growth
        tolerance_grad (float): termination tolerance on first order optimality
        tolerance_change (float): termination tolerance on function value/parameter changes
        max_iter (int): maximal number of iterations per optimization step
        line_search_fn (str): 'strong' or None

    """

    def __init__(self, params, lr=1, history_size=10, norm_growth_bound=2,
                 initial_magnitude = 0.01, lr_pos = 1.1, lr_neg = 0.9):

        defaults = dict(lr=lr, history_size=history_size, norm_growth_bound=norm_growth_bound)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("BoundedGrowthLBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._lr = lr  # Kept for consistency, but standard L-BFGS doesn't use LR directly
        self._history_size = history_size
        self._norm_growth_bound = norm_growth_bound

        self.lr_pos = lr_pos
        self.lr_neg = lr_neg

        self.state['step'] = 0
        self.state['H_diag'] = 1  # Initial Hessian diagonal approximation
        self.state['s_history'] = []
        self.state['y_history'] = []
        self.state['previous_update_magnitude'] = {} # Store previous update magnitude for each parameter

        for p in self._params:
            self.state['previous_update_magnitude'][p] = torch.full_like(p, initial_magnitude) # Initialize to zero


    def _update_history(self, s, y):
        """Updates history buffer with new s and y."""
        self.state['s_history'].append(s)
        self.state['y_history'].append(y)
        if len(self.state['s_history']) > self._history_size:
            self.state['s_history'].pop(0)
            self.state['y_history'].pop(0)

    def _directional_evaluate(self, closure, t, x, d):
        """Evaluates the function and gradient at x + t*d."""
        self._set_param(x + t * d)
        f, g = self._eval_closure(closure)
        return f, g

    def _set_param(self, param_vec):
        """Sets the parameters to param_vec."""
        current_index = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.data.copy_(param_vec[current_index:current_index + numel].view_as(p))
                current_index += numel

    def _gather_flat_param(self):
        """Gather the parameters into a single flat vector."""
        views = []
        for group in self.param_groups:
            for p in group['params']:
                view = p.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_grad(self):
        """Gather the gradients into a single flat vector."""
        views = []
        for group in self.param_groups:
            for p in group['params']:
                view = p.grad.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    def _eval_closure(self, closure, flat_param=None):
        """Evaluates the closure and returns function value and gradient."""
        if flat_param is not None:
            self._set_param(flat_param)
        return closure() # closure should return loss, grad

    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step."""
        assert len(self.param_groups) == 1, "LBFGS doesn't support multiple param groups"

        group = self.param_groups[0]
        params = group['params']

        state = self.state
        state['step'] += 1

        # Evaluate initial f(x) and g(x)
        with torch.enable_grad(): loss = closure()

        # Flatten parameters and gradients
        flat_param = self._gather_flat_param()
        flat_grad = self._gather_flat_grad()
        prev_flat_param = flat_param.clone()

        # History buffers
        s_history = state.get('s_history')
        y_history = state.get('y_history')
        H_diag = state.get('H_diag')

        # L-BFGS Direction Calculation (Two-Loop Recursion)
        rho = []
        alpha = []

        q = flat_grad.clone()
        for s, y in zip(reversed(s_history), reversed(y_history)):
            rho_i = 1.0 / torch.dot(y, s)
            rho.append(rho_i)
            alpha_i = rho_i * torch.dot(s, q)
            alpha.append(alpha_i)
            q.add_(-alpha_i, y)

        # Multiply by initial Hessian approximation (H0 = gamma * I, where gamma = (s_k^T y_k) / (y_k^T y_k))
        gamma = 1.0
        if y_history: # Use gamma scaling if history exists
            s_k = s_history[-1]
            y_k = y_history[-1]
            gamma = torch.dot(s_k, y_k) / torch.dot(y_k, y_k) if torch.dot(y_k, y_k) > 0 else 1.0
        r = torch.mul(H_diag * gamma, q)

        for s, y, alpha_i in zip(s_history, y_history, reversed(alpha)):
            beta_i = rho.pop() * torch.dot(y, r)
            r.add_(s, alpha = (alpha_i - beta_i))

        direction = -r * self._lr # Search direction (unclipped update)

        # Element-wise Bounded Update Growth Clipping
        clipped_direction = direction.clone()
        current_index = 0
        for p in params:
            numel = p.numel()
            param_direction = direction[current_index:current_index + numel].view_as(p)
            prev_update_mag = state['previous_update_magnitude'][p]

            clipped_param_direction = torch.clip(param_direction,
                                                 -prev_update_mag * self._norm_growth_bound,
                                                  prev_update_mag * self._norm_growth_bound)

            clipped_direction[current_index:current_index + numel] = clipped_param_direction.view(-1)
            state['previous_update_magnitude'][p] = torch.abs(clipped_param_direction) # Update magnitude for next step
            current_index += numel


        # Update parameters with clipped direction
        flat_param.add_(clipped_direction)
        self._set_param(flat_param)

        # Evaluate new loss and grad
        with torch.enable_grad(): new_loss = closure()
        if new_loss > loss: self._lr *= self.lr_neg
        else: self._lr *= self.lr_pos
        new_grad_flat = self._gather_flat_grad()

        # Update history (using the CLIPED update for s_k and unclipped gradient difference for y_k)
        s_k = flat_param - prev_flat_param # Use clipped update as 's'
        y_k = new_grad_flat - flat_grad      # Use unclipped grad difference as 'y'
        self._update_history(s_k, y_k)

        return new_loss