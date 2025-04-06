# pylint:disable=signature-differs, not-callable

import math

import torch


class DirectionalExp(torch.optim.Optimizer):
    """basically like directional newton step but with an exponential model instead of quadratic, I actually think this is better."""
    def __init__(self, params, lr=1e-3, delta=1e-3, val_step = False):
        defaults = dict(lr=lr, delta=delta)
        super().__init__(params, defaults)
        self.val_step=val_step

    @torch.no_grad
    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for DirectionalExp")

        # Initial loss and gradient computation
        with torch.enable_grad():
            loss = closure()

        # Save current parameters and compute gradient direction
        current_params = []
        grad_direction = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                current_params.append(p.detach().clone())
                grad_dir = -p.grad.data.clone()
                grad_direction.append(grad_dir)

        # Perturb parameters by +delta and compute loss_plus
        idx = 0
        for group in self.param_groups:
            delta_val = group['delta']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(grad_direction[idx], alpha=delta_val)
                idx += 1

        loss_plus = closure(False)

        # Restore parameters, perturb by -delta and compute loss_minus
        idx = 0
        for group in self.param_groups:
            delta_val = group['delta']
            for p in group['params']:
                if p.grad is None: # grad is from 1st step
                    continue
                p.data.add_(grad_direction[idx], alpha=-delta_val*2)
                idx += 1

        loss_minus = closure(False)

        # Restore parameters again
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: # grad is from 1st step
                    continue
                p.data.copy_(current_params[idx])
                idx += 1

        # Fit exponential model and compute step size
        delta_val = self.param_groups[0]['delta']
        loss0 = loss.item()
        lp = loss_plus.item()
        lm = loss_minus.item()

        e_d = math.exp(delta_val)
        e_neg_d = math.exp(-delta_val)
        a11 = e_d - 1
        a12 = e_neg_d - 1
        a21 = a12
        a22 = a11
        det = a11 * a22 - a12 * a21

        if det == 0:
            alpha = self.param_groups[0]['lr']
        else:
            # Solve for b and c
            b_numerator = (lp - loss0) * a22 - (lm - loss0) * a12
            c_numerator = (lm - loss0) * a11 - (lp - loss0) * a21
            b = b_numerator / det
            c = c_numerator / det

            if b > 0 and c > 0:
                alpha = 0.5 * math.log(c / b)
                # Clamp alpha to prevent excessively large steps
                alpha = max(min(alpha, 1.0 / self.param_groups[0]['delta']), -1.0 / self.param_groups[0]['delta'])
            else:
                alpha = self.param_groups[0]['lr']

        # Update parameters with computed alpha
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(grad_direction[idx], alpha=alpha)
                idx += 1

        if self.val_step:
            final_loss = closure(False)
            if final_loss > loss and alpha != self.param_groups[0]['lr']:
                # undo update
                idx = 0
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue

                        p.data.sub_(grad_direction[idx], alpha=alpha - self.param_groups[0]['lr'])
                        idx += 1

        return loss