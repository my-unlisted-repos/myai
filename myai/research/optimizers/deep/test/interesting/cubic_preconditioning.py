# pylint:disable=signature-differs, not-callable

import math

import torch
from torch.optim import Optimizer


class CubicPreconditioning(Optimizer):
    """I would set line_search_freq to None to disable it, or set it to 3 to always use it (keep some iters for model building). This makes 2 completely different methods but both may work well, but they haven't so far"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.99), epsilon=1e-6,
                 update_freq=1, line_search_freq:int|None=100, line_search_iters=10, line_search_init_lr = 1,
                 cubic_solve_eps=1e-8, max_ls=5, max_third_order=1.0,
                 h_floor=1e-8, t_floor=1e-12):
        defaults = dict(lr=lr, betas=betas, epsilon=epsilon,
                        update_freq=update_freq, line_search_freq=line_search_freq,
                        line_search_iters=line_search_iters,
                        cubic_solve_eps=cubic_solve_eps, max_ls=max_ls, max_third_order=max_third_order, h_floor=h_floor,
                        t_floor=t_floor)
        super().__init__(params, defaults)
        self.step_count = 0
        self.line_search_lr = line_search_init_lr

    @torch.no_grad
    def step(self, closure=None):
        line_search_freq = self.defaults['line_search_freq']
        if line_search_freq is None:
            return self._regular_step(closure)

        if closure is None and self.step_count % line_search_freq == 0:
            raise ValueError("Closure required for line search steps")

        if self.step_count != 0 and self.step_count % line_search_freq == 0:
            return self._third_order_step(closure)
        return self._regular_step(closure)

    @torch.no_grad
    def _regular_step(self, closure):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Initialize state
                if 'step' not in state:
                    state['step'] = 0
                    state['prev_grad'] = torch.zeros_like(p)
                    state['prev_update'] = torch.zeros_like(p)
                    state['H'] = torch.ones_like(p)  # Diagonal Hessian approx
                    state['T'] = torch.zeros_like(p)  # Diagonal third-order term

                state['step'] += 1
                prev_grad = state['prev_grad']
                prev_update = state['prev_update']

                if state['step'] % group['update_freq'] == 0:
                    s = prev_update
                    delta_grad = grad - prev_grad

                    # Stabilized Hessian update with floor
                    H_prev = state['H'].clone()
                    state['H'] = group['betas'][0] * H_prev + (1 - group['betas'][0]) * delta_grad.pow(2)
                    state['H'] = torch.max(state['H'], torch.full_like(state['H'], group['h_floor']))

                    # Robust third-order estimation
                    with torch.no_grad():
                        H_hat = state['H'] / (1 - group['betas'][0]**state['step'])
                        residual = delta_grad - H_hat * s

                        # Adaptive denominator stabilization
                        denom = s.abs() * H_hat.sqrt() + group['epsilon']
                        denom = torch.max(denom, torch.full_like(denom, group['t_floor']))

                        # Scaled third-order estimation
                        T_est = residual / denom

                        # Adaptive clipping based on parameter scale
                        param_scale = p.data.abs().mean().item()
                        max_T = group['max_third_order'] / (param_scale + 1e-6)
                        T_est = torch.clamp(T_est, -max_T, max_T)

                        # Heavy momentum for third-order terms
                        state['T'] = group['betas'][2] * state['T'] + (1 - group['betas'][2]) * T_est

                # Modified preconditioned update with stabilized third-order term
                H = state['H'] / (1 - group['betas'][0]**state['step'])
                T = state['T'] / (1 - group['betas'][2]**state['step'])

                # Scale third-order term by learning rate squared
                lr_sq = group['lr'] ** 2
                precond_grad = grad / (H.sqrt() + group['epsilon'])
                third_order = 0.5 * lr_sq * T * precond_grad.pow(2)

                # Final update with gradient clipping
                update = -group['lr'] * (precond_grad + third_order)
                update = torch.clamp(update, -10*group['lr'], 10*group['lr'])

                state['prev_grad'] = grad.clone()
                state['prev_update'] = update.clone()
                p.add_(update)

        self.step_count += 1
        return loss

    @torch.no_grad
    def _third_order_step(self, closure=None):
        if closure is None:
            return self._regular_step(closure)

        group = self.param_groups[0]
        max_ls = group['max_ls']
        cubic_eps = group['cubic_solve_eps']
        lr = group['lr']

        # Initial loss and gradient
        with torch.enable_grad(): loss = closure()
        orig_loss = loss.item()
        params = [p for g in self.param_groups for p in g['params'] if p.grad is not None]
        orig_params = [p.clone() for p in params]
        grads = [p.grad.clone() for p in params]

        # Compute preconditioned direction
        dir = []
        for p in params:
            state = self.state[p]
            H = state['H'] / (1 - group['betas'][0]**state['step'])
            T = state['T'] / (1 - group['betas'][1]**state['step'])
            g = p.grad.data
            precond_g = g / (H.sqrt() + group['epsilon'])
            third_order = 0.5 * T * precond_g**2
            d = -(precond_g + third_order)
            dir.append(d)

        # Determine cubic coefficients
        b = sum((g * d).sum() for g, d in zip(grads, dir)).item() # type:ignore
        a = 0.5 * sum((state['T'] * d**3).sum().item()
                      for p, d in zip(params, dir)
                      for state in [self.state[p]])

        # Solve cubic equation: a*α² + b*α + c = 0 (c=0 since initial step is α=0)
        alpha1, alpha2 = 0, 0
        if abs(a) > cubic_eps:
            discr = b**2
            sqrt_discr = math.sqrt(max(discr, 0))
            alpha1 = (-b + sqrt_discr) / (2 * a)
            alpha2 = (-b - sqrt_discr) / (2 * a)
        else:
            alpha1 = -b / (2 * a) if a != 0 else lr

        # Evaluate candidates
        best_loss = orig_loss
        best_alpha = 0

        cur = self.line_search_lr
        lrs = [alpha1, alpha2, cur]
        for _ in range(max_ls):
            cur /= 2
            lrs.extend([cur, -cur])

        for alpha in lrs:
            if alpha <= 0:
                continue
            # Update parameters
            for p, o, d in zip(params, orig_params, dir):
                p.copy_(o + alpha * d)  # Simplified for example
            with torch.no_grad():
                new_loss = closure(False)
            if new_loss < best_loss:
                best_loss = new_loss
                best_alpha = alpha
            # Restore parameters
            for p, orig in zip(params, orig_params):
                p.copy_(orig)

        # Apply best step
        if best_alpha != 0:
            for p, d in zip(params, dir):
                p.add_(best_alpha * d)
            self.line_search_lr *= 2
        else:
            # Fallback to regular step
            self._regular_step(closure)
            self.line_search_lr /= 2

        self.step_count += 1
        return best_loss if best_alpha != 0 else loss