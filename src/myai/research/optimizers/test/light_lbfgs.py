# pylint:disable=signature-differs, not-callable

from collections import defaultdict

import torch
from torch.optim import Optimizer


class LightLBFGS(Optimizer):
    """just lbfgs but need to tests against pytorch"""
    def __init__(self, params, lr=1.0, history_size=5, max_iter=50,
                 tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn='backtracking'):
        defaults = dict(lr=lr, history_size=history_size, max_iter=max_iter,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        line_search_fn=line_search_fn)
        super().__init__(params, defaults)
        self._state = defaultdict(dict)

        for group in self.param_groups:
            state = self._state[0]
            state['func_evals'] = 0
            state['n_iter'] = 0
            state['t'] = lr
            state['prev_flat_grad'] = None
            state['H_diag'] = 1.0
            state['s_history'] = []
            state['y_history'] = []
            state['rho_history'] = []

    def _gather_flat_grad(self, group):
        views = []
        for p in group['params']:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_params(self, group):
        views = []
        for p in group['params']:
            views.append(p.view(-1))
        return torch.cat(views, 0)

    def _distribute_flat_params(self, group, flat_params):
        offset = 0
        for p in group['params']:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset+numel].view_as(p))
            offset += numel

    def _add_grad(self, group, step_size, direction):
        offset = 0
        for p in group['params']:
            numel = p.numel()
            p.data.add_(direction[offset:offset+numel].view_as(p), alpha=step_size)
            offset += numel

    def step(self, closure):
        group = self.param_groups[0]
        state = self._state[0]
        lr = group['lr']
        max_iter = group['max_iter']
        line_search_fn = group['line_search_fn']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']

        # Initial function evaluation
        orig_loss = closure()
        loss = orig_loss.item()
        current_flat_params = self._gather_flat_params(group)
        current_flat_grad = self._gather_flat_grad(group)

        # Check convergence
        g_norm = current_flat_grad.norm()
        if g_norm <= tolerance_grad:
            return orig_loss

        # L-BFGS direction computation
        s_history = state['s_history']
        y_history = state['y_history']
        rho_history = state['rho_history']
        H_diag = state['H_diag']

        q = current_flat_grad.clone()
        alpha = []

        # Two-loop recursion for inverse Hessian approximation
        for i in reversed(range(len(s_history))):
            alpha_i = rho_history[i] * s_history[i].dot(q)
            alpha.append(alpha_i)
            q.add_(y_history[i], alpha=-alpha_i.detach().cpu().item())

        r = q * H_diag

        for i in range(len(s_history)):
            beta = rho_history[i] * y_history[i].dot(r)
            r.add_(s_history[i], alpha=(alpha[len(alpha)-1-i] - beta).detach().cpu().item())

        direction = r.neg()

        # Line search implementation
        if line_search_fn == "strong_wolfe":
            t, ls_func_evals = self._strong_wolfe(
                closure, current_flat_params, current_flat_grad,
                loss, direction, orig_loss, group
            )
        else:
            t, ls_func_evals = self._backtracking(
                closure, current_flat_params, current_flat_grad,
                loss, direction, group
            )

        state['func_evals'] += ls_func_evals

        # Update parameters
        self._add_grad(group, t, direction)
        new_flat_params = self._gather_flat_params(group)
        new_loss = closure()
        new_flat_grad = self._gather_flat_grad(group)

        # Compute differences
        s = new_flat_params - current_flat_params
        y = new_flat_grad - current_flat_grad
        ys = y.dot(s)

        # Update history
        if ys > 1e-10:
            if len(s_history) >= group['history_size']:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)

            s_history.append(s)
            y_history.append(y)
            rho_history.append(1.0 / ys)

            # Update Hessian diagonal
            yy = y.dot(y)
            H_diag = ys / yy
            state['H_diag'] = H_diag

        # Check convergence
        param_change = new_flat_params - current_flat_params
        if param_change.norm() < tolerance_change:
            return new_loss

        if new_flat_grad.norm() < tolerance_grad:
            return new_loss

        return new_loss

    def _backtracking(self, closure, x_init, g_init, f_init, direction, group):
        t = group['lr']
        dec = 0.5
        max_ls = 20
        min_step = 1e-20

        for i in range(max_ls):
            self._distribute_flat_params(group, x_init + t * direction)
            with torch.enable_grad():
                f_new = closure().item()

            if f_new < f_init + 1e-4 * t * g_init.dot(direction):
                return t, i+1
            t *= dec
            if t < min_step:
                break
        return t, max_ls

    def _strong_wolfe(self, closure, x_init, g_init, f_init, direction, orig_loss, group):
        t = group['lr']
        c1 = 1e-4
        c2 = 0.9
        max_ls = 20
        min_step = 1e-20

        f_prev = torch.tensor(f_init)
        g_prev = g_init.clone()
        t_prev = torch.tensor(0.0)
        t_cur = torch.tensor(t)

        for i in range(max_ls):
            self._distribute_flat_params(group, x_init + t_cur * direction)
            with torch.enable_grad():
                f_cur = closure().item()
                g_cur = self._gather_flat_grad(group)

            gtd = g_cur.dot(direction)
            if f_cur > f_init + c1 * t_cur * g_init.dot(direction) or (i > 0 and f_cur >= f_prev):
                return self._zoom(closure, x_init, g_init, f_init, direction,
                                 t_prev, t_cur, f_prev, f_cur, g_prev, g_cur, group)

            if abs(gtd) <= -c2 * g_init.dot(direction):
                return t_cur, i+1

            if gtd >= 0:
                return self._zoom(closure, x_init, g_init, f_init, direction,
                                 t_cur, t_prev, f_cur, f_prev, g_cur, g_prev, group)

            f_prev = f_cur
            g_prev = g_cur
            t_prev = t_cur
            t_cur *= 2

        return t_cur, max_ls

    def _zoom(self, closure, x_init, g_init, f_init, direction,
             t_lo, t_hi, f_lo, f_hi, g_lo, g_hi, group):
        c1 = 1e-4
        c2 = 0.9
        max_iter = 20

        for _ in range(max_iter):
            t = (t_lo + t_hi) / 2
            self._distribute_flat_params(group, x_init + t * direction)
            with torch.enable_grad():
                f = closure().item()
                g = self._gather_flat_grad(group)

            gtd = g.dot(direction)
            if f > f_init + c1 * t * g_init.dot(direction) or f >= f_lo:
                t_hi = t
                f_hi = f
                g_hi = g
            else:
                if abs(gtd) <= -c2 * g_init.dot(direction):
                    return t, 0
                if gtd * (t_hi - t_lo) >= 0:
                    t_hi = t_lo
                t_lo = t
                f_lo = f
                g_lo = g

            if abs(t_hi - t_lo) < 1e-9:
                break

        return t, 0 # type:ignore