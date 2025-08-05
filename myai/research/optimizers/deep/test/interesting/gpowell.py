# pylint:disable=signature-differs, not-callable
import torch
from torch.optim import Optimizer

class GPowell(Optimizer):
    """powell that uses gradient information, but as it seems not very efficiently"""
    def __init__(self, params, c1=1e-4, c2=0.4, max_ls=20, max_zoom=20,
                direction_history=5, reset_threshold=0.1):
        defaults = dict(c1=c1, c2=c2, max_ls=max_ls, max_zoom=max_zoom,
                      direction_history=direction_history, reset_threshold=reset_threshold)
        super().__init__(params, defaults)

        self.global_state = {
            'prev_directions': [],
            'prev_loss': None,
            'prev_params': None,
            'prev_grad': None,
            'iteration': 0
        }

        self._initialized = False  # Track initialization state
        self.c1 = c1
        self.c2 = c2

    def _get_flat_gradient(self, params):
        return torch.cat([p.grad.view(-1) for p in params if p.grad is not None])

    def _get_flat_params(self, params):
        return torch.cat([p.data.view(-1) for p in params])

    def _set_flat_params(self, params, flat_tensor):
        offset = 0
        for p in params:
            numel = p.numel()
            p.data.copy_(flat_tensor[offset:offset+numel].view_as(p.data))
            offset += numel

    def _add_direction(self, new_dir, group):
        """Maintain direction history with curvature information"""
        if len(group['directions']) >= group['direction_history']:
            group['directions'].pop(0)

        # Orthogonalize with previous directions
        new_dir = new_dir.clone()
        for d in group['directions']:
            new_dir -= d * (d.dot(new_dir) / (d.dot(d) + 1e-8))

        norm = torch.norm(new_dir)
        if norm > 1e-8:
            new_dir /= norm
        group['directions'].append(new_dir)

    def step(self, closure):
        closure = torch.enable_grad()(closure)
        group = self.param_groups[0]
        params = group['params']

        # Initialization on first step
        if not self._initialized:
            with torch.enable_grad():
                loss = closure()
                # loss.backward()
                initial_grad = self._get_flat_gradient(params)
                group['directions'] = [initial_grad/torch.norm(initial_grad)]
                self._initialized = True

        # Initial evaluation
        loss = closure()
        # loss.backward()
        current_grad = self._get_flat_gradient(params)
        current_params = self._get_flat_params(params)

        # Store initial state
        if self.global_state['prev_grad'] is None:
            self.global_state.update({
                'prev_loss': loss.item(),
                'prev_params': current_params.clone(),
                'prev_grad': current_grad.clone()
            })
            return loss

        # Update search directions with gradient information
        delta_grad = current_grad - self.global_state['prev_grad']
        delta_params = current_params - self.global_state['prev_params']

        # Curvature-aware direction update (BFGS-style)
        if torch.abs(delta_grad.dot(delta_params)) > 1e-8:
            new_dir = self._compute_bfgs_direction(delta_grad, delta_params, current_grad)
            self._add_direction(new_dir, group)

        # Perform line search in each direction
        for direction in group['directions']:
            self._curvature_aware_line_search(closure, direction, params)

        # Update state for next iteration
        self.global_state.update({
            'prev_loss': loss.item(),
            'prev_params': self._get_flat_params(params).clone(),
            'prev_grad': current_grad.clone(),
            'iteration': self.global_state['iteration'] + 1
        })

        return loss

    def _compute_bfgs_direction(self, delta_grad, delta_params, current_grad):
        """Compute BFGS-inspired search direction"""
        y_dot_s = delta_grad.dot(delta_params)
        if y_dot_s < 1e-8:
            return -current_grad

        rho = 1.0 / y_dot_s
        I = torch.eye(len(delta_grad), device=delta_grad.device)

        # BFGS update formula
        V = I - rho * torch.outer(delta_params, delta_grad)
        H_k = torch.mm(V.t(), torch.mm(I, V)) + rho * torch.outer(delta_params, delta_params)

        return -torch.mv(H_k, current_grad)

    def _curvature_aware_line_search(self, closure, direction, params):
        group = self.param_groups[0]
        original_params = self._get_flat_params(params)
        initial_loss = self.global_state['prev_loss']
        current_grad = self.global_state['prev_grad']

        # Convert direction to parameter space
        dir_tensor = direction.to(original_params.device)
        gd = current_grad.dot(dir_tensor)

        if gd >= 0:  # Ensure descent direction
            dir_tensor = -current_grad # pylint:disable = invalid-unary-operand-type
            gd = current_grad.dot(dir_tensor)

        alpha, _ = self._strong_wolfe(
            lambda a: self._eval_closure(closure, original_params, dir_tensor, a),
            initial_loss, gd, self.c1, self.c2, group['max_ls'], group['max_zoom']
        )

        # Update parameters with found alpha
        new_params = original_params + alpha * dir_tensor
        self._set_flat_params(params, new_params)

    def _eval_closure(self, closure, base_params, direction, alpha):
        self._set_flat_params(self.param_groups[0]['params'], base_params + alpha * direction)
        with torch.enable_grad():
            loss = closure()
            # loss.backward()
        flat_grad = self._get_flat_gradient(self.param_groups[0]['params'])
        g_dir = torch.dot(flat_grad, direction).item()
        return loss.item(), g_dir

    def _strong_wolfe(self, func, f0, g0, c1, c2, max_ls=20, max_zoom=30):
        alpha = 1.0
        alpha_prev = 0.0
        f_prev = f0
        g_prev = g0

        for _ in range(max_ls):
            f, g = func(alpha)
            if f > f0 + c1 * alpha * g0 or (f >= f_prev and _ > 0):
                return self._zoom(func, alpha_prev, alpha, f_prev, f, g_prev, g, f0, g0, c1, c2)

            if abs(g) <= -c2 * g0:
                return alpha, f

            if g >= 0:
                return self._zoom(func, alpha_prev, alpha, f_prev, f, g_prev, g, f0, g0, c1, c2)

            alpha_prev = alpha
            f_prev = f
            g_prev = g
            alpha *= 2

        return alpha_prev, f_prev

    def _zoom(self, func, lo, hi, f_lo, f_hi, g_lo, g_hi, f0, g0, c1, c2):
        for _ in range(self.param_groups[0]['max_zoom']):
            alpha = (lo + hi) / 2
            f, g = func(alpha)

            if f > f0 + c1 * alpha * g0 or f >= f_lo:
                hi = alpha
                f_hi = f
                g_hi = g
            else:
                if abs(g) <= -c2 * g0:
                    return alpha, f

                if g * (hi - lo) >= 0:
                    hi = lo
                lo = alpha
                f_lo = f
                g_lo = g

        return alpha, f # type:ignore