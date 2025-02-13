# pylint:disable=signature-differs, not-callable
"""DeepSeek"""
import math

import torch
from torch.optim import Optimizer


class PoincareThreshold(Optimizer):
    """performs poincare something when loss decreases by threshold_ratio."""
    def __init__(self, params, lr=1e-3, threshold_ratio=0.1, momentum=0.9, curvature_scale=0.1):
        params = list(params)
        if any(isinstance(p, dict) for p in params): raise NotImplementedError('PoincareThreshold doesnt support param groups')

        defaults = dict(lr=lr, threshold_ratio=threshold_ratio, momentum=momentum, curvature_scale=curvature_scale)
        super().__init__(params, defaults)

        self._state = {
            'previous_loss': None,
            'previous_params': None,
            'previous_grads': None,
            'angle_buffer': [],
            'update_buffer': [],
            'section_crossed': False
        }
    @torch.no_grad
    def step(self, closure):
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        else:
            raise ValueError("A closure must be provided to compute the loss.")

        group = self.param_groups[0]
        lr = group['lr']
        threshold_ratio = group['threshold_ratio']
        momentum = group['momentum']
        curvature_scale = group['curvature_scale']

        # Initialize state on first step
        if self._state['previous_loss'] is None:
            self._state['previous_loss'] = loss.item()
            self._state['previous_params'] = [p.detach().clone() for p in group['params']]
            self._state['previous_grads'] = [p.grad.detach().clone() for p in group['params']]
            return

        # Check if Poincaré section is crossed (loss decrease > threshold)
        loss_decrease = self._state['previous_loss'] - loss.item()
        threshold = self._state['previous_loss'] * threshold_ratio
        self._state['section_crossed'] = loss_decrease >= threshold

        if self._state['section_crossed']:
            # Analyze trajectory between sections
            current_grads = [p.grad.detach().clone() for p in group['params']]
            current_params = [p.detach().clone() for p in group['params']]

            # Compute angle between previous and current gradients (stability indicator)
            grad_dot = sum(
                (g_prev.flatten() @ g_curr.flatten()).item()
                for g_prev, g_curr in zip(self._state['previous_grads'], current_grads)
            )
            grad_norm_prev = math.sqrt(sum(
                torch.norm(g_prev).item() ** 2 for g_prev in self._state['previous_grads'] # pylint:disable=not-an-iterable
            ))
            grad_norm_curr = math.sqrt(sum(
                torch.norm(g_curr).item() ** 2 for g_curr in current_grads
            ))
            t = grad_dot / (grad_norm_prev * grad_norm_curr + 1e-8) - 1e-8
            if t > 1: t = 1
            angle = math.acos(t)

            # Compute parameter displacement since last section
            delta_params = [
                p_curr - p_prev for p_curr, p_prev in zip(current_params, self._state['previous_params'])
            ]
            delta_norm = sum(torch.norm(delta).item() ** 2 for delta in delta_params) ** 0.5

            # Reshape the update direction using curvature and angle
            # Rule: If angle > 90 degrees (oscillation), project gradient onto previous displacement
            #       If angle < 90 degrees (stable), amplify the update along the displacement
            for p, delta, grad in zip(group['params'], delta_params, current_grads):
                if delta_norm > 0:
                    if angle > math.pi / 2:  # Oscillation detected
                        # Project gradient onto the displacement direction to dampen oscillations
                        proj_coeff = (grad.flatten() @ delta.flatten()) / (delta_norm ** 2 + 1e-8)
                        reshaped_grad = proj_coeff * delta
                    else:  # Stable direction
                        # Amplify the gradient along the displacement direction
                        reshaped_grad = grad + curvature_scale * delta / delta_norm

                    # Apply momentum and update
                    if 'momentum_buffer' not in self.state[p]:
                        self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(momentum).add_(reshaped_grad, alpha=lr)
                    p.data.add_(-buf)
                else:
                    p.data.add_(-lr, grad)

            # Update state for next section
            self._state['previous_loss'] = loss.item()
            self._state['previous_params'] = current_params
            self._state['previous_grads'] = current_grads
        else:
            # Standard SGD update if not crossing a section
            for p in group['params']:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha = -lr)



class PoincareDualThreshold(Optimizer):
    """same as previous one, but does two steps per batch and does poincare something if second loss decreased compared to first one by threshold ratio."""
    def __init__(self, params, lr=1e-3, threshold_ratio=1e-12, momentum=0.9, curvature_scale=0.1):
        params = list(params)
        if any(isinstance(p, dict) for p in params): raise NotImplementedError('PoincareDualThreshold doesnt support param groups')

        defaults = dict(lr=lr, threshold_ratio=threshold_ratio, momentum=momentum, curvature_scale=curvature_scale)
        super().__init__(params, defaults)

        self._state = {
            'previous_loss': None,
            'previous_params': None,
            'previous_grads': None,
            'angle_buffer': [],
            'update_buffer': [],
            'section_crossed': False
        }

    def _reset_state(self):
        self._state = {
            'previous_loss': None,
            'previous_params': None,
            'previous_grads': None,
            'angle_buffer': [],
            'update_buffer': [],
            'section_crossed': False
        }

    @torch.no_grad
    def step(self, closure):
        self._reset_state()
        self._step(closure)
        return self._step(closure)

    @torch.no_grad
    def _step(self, closure):
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        else:
            raise ValueError("A closure must be provided to compute the loss.")

        group = self.param_groups[0]
        lr = group['lr']
        threshold_ratio = group['threshold_ratio']
        momentum = group['momentum']
        curvature_scale = group['curvature_scale']

        # Initialize state on first step
        if self._state['previous_loss'] is None:
            self._state['previous_loss'] = loss.item()
            self._state['previous_params'] = [p.detach().clone() for p in group['params']]
            self._state['previous_grads'] = [p.grad.detach().clone() for p in group['params']]
            return

        # Check if Poincaré section is crossed (loss decrease > threshold)
        loss_decrease = self._state['previous_loss'] - loss.item()
        threshold = self._state['previous_loss'] * threshold_ratio
        self._state['section_crossed'] = loss_decrease >= threshold

        if self._state['section_crossed']:
            # Analyze trajectory between sections
            current_grads = [p.grad.detach().clone() for p in group['params']]
            current_params = [p.detach().clone() for p in group['params']]

            # Compute angle between previous and current gradients (stability indicator)
            grad_dot = sum(
                (g_prev.flatten() @ g_curr.flatten()).item()
                for g_prev, g_curr in zip(self._state['previous_grads'], current_grads)
            )
            grad_norm_prev = math.sqrt(sum(
                torch.norm(g_prev).item() ** 2 for g_prev in self._state['previous_grads'] # pylint:disable=not-an-iterable
            ))
            grad_norm_curr = math.sqrt(sum(
                torch.norm(g_curr).item() ** 2 for g_curr in current_grads
            ))
            t = grad_dot / (grad_norm_prev * grad_norm_curr + 1e-8) - 1e-8
            if t > 1: t = 1
            angle = math.acos(t)

            # Compute parameter displacement since last section
            delta_params = [
                p_curr - p_prev for p_curr, p_prev in zip(current_params, self._state['previous_params'])
            ]
            delta_norm = sum(torch.norm(delta).item() ** 2 for delta in delta_params) ** 0.5

            # Reshape the update direction using curvature and angle
            # Rule: If angle > 90 degrees (oscillation), project gradient onto previous displacement
            #       If angle < 90 degrees (stable), amplify the update along the displacement
            for p, delta, grad in zip(group['params'], delta_params, current_grads):
                if delta_norm > 0:
                    if angle > math.pi / 2:  # Oscillation detected
                        # Project gradient onto the displacement direction to dampen oscillations
                        proj_coeff = (grad.flatten() @ delta.flatten()) / (delta_norm ** 2 + 1e-8)
                        reshaped_grad = proj_coeff * delta
                    else:  # Stable direction
                        # Amplify the gradient along the displacement direction
                        reshaped_grad = grad + curvature_scale * delta / delta_norm

                    # Apply momentum and update
                    if 'momentum_buffer' not in self.state[p]:
                        self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(momentum).add_(reshaped_grad, alpha=lr)
                    p.data.add_(-buf)
                else:
                    p.data.add_(grad, alpha = -lr, )

            # Update state for next section
            self._state['previous_loss'] = loss.item()
            self._state['previous_params'] = current_params
            self._state['previous_grads'] = current_grads
        else:
            # Standard SGD update if not crossing a section
            for p in group['params']:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha = -lr)



class PoincareDualStep(Optimizer):
    """another versions =="""
    def __init__(self, params, lr=1e-3, momentum=0.9, curvature_scale=0.1, stability_epsilon=1e-6, dual_step=True):
        params = list(params)
        if any(isinstance(p, dict) for p in params): raise NotImplementedError('PoincareDualStep doesnt support param groups')

        defaults = dict(lr=lr, momentum=momentum,
                       curvature_scale=curvature_scale,
                       stability_epsilon=stability_epsilon)
        super().__init__(params, defaults)

        # Poincaré system state
        self.poincare_state = {
            'phase': 0,
            'original_params': None,
            'original_grads': None,
            'trajectory_buffer': [],
            'convergence_factor': 1.0
        }

        self.dual_step = dual_step

        # Initialize momentum buffers in PyTorch's native state
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    @torch.no_grad
    def step(self, closure):
        if self.dual_step:
            assert self.poincare_state['phase'] == 0
            self._step(closure)
            return self._step(closure)

        return self._step(closure)

    @torch.no_grad
    def _step(self, closure):
        """Dual-phase optimization with enhanced Poincaré dynamics"""
        group = self.param_groups[0]
        lr = group['lr']
        momentum = group['momentum']
        c_scale = group['curvature_scale']
        eps = group['stability_epsilon']

        if self.poincare_state['phase'] == 0:
            # Phase 1: Tentative update and state capture --------------------------------
            with torch.enable_grad(): loss = closure()

            # Capture current system state
            self.poincare_state['original_params'] = [p.detach().clone() for p in group['params']]
            self.poincare_state['original_grads'] = [p.grad.detach().clone() for p in group['params']]

            # Apply tentative update with momentum
            for p in group['params']:
                if p.grad is None:
                    continue
                buf = self.state[p]['momentum_buffer']
                buf.mul_(momentum).add_(p.grad, alpha=lr)
                p.data.sub_(buf)

            self.poincare_state['phase'] = 1
            return loss

        # Phase 2: Poincaré-adjusted update ------------------------------------------
        with torch.enable_grad(): loss = closure()

        # Capture post-update state
        updated_params = [p.detach().clone() for p in group['params']]
        updated_grads = [p.grad.detach().clone() for p in group['params']]

        # Compute trajectory dynamics
        delta_params = [up - op for up, op in zip(updated_params,
                                                    self.poincare_state['original_params'])]
        delta_norm = sum(torch.norm(d).item()**2 for d in delta_params)**0.5

        # Enhanced stability-preserved cosine similarity
        grad_dot = sum(
            (g_orig.flatten() @ g_upd.flatten()).item()
            for g_orig, g_upd in zip(self.poincare_state['original_grads'], updated_grads)
        )
        grad_norms = (
            sum(torch.norm(g).item()**2 for g in self.poincare_state['original_grads'])**0.5, # pylint:disable=not-an-iterable
            sum(torch.norm(g).item()**2 for g in updated_grads)**0.5
        )

        # Numerically robust angle computation
        with torch.no_grad():
            cos_sim = grad_dot / (grad_norms[0] * grad_norms[1] + eps)
            cos_sim = torch.clamp(torch.tensor(cos_sim), -1.0, 1.0).item()
        angle = math.acos(cos_sim)

        # Adaptive curvature scaling based on trajectory consistency
        if len(self.poincare_state['trajectory_buffer']) >= 2:
            prev_angle = self.poincare_state['trajectory_buffer'][-1]['angle']
            angle_diff = abs(angle - prev_angle)
            c_scale *= 1.0 + math.exp(-angle_diff)  # Adaptive scaling

        # Store trajectory information (limited history)
        self.poincare_state['trajectory_buffer'].append({
            'delta_norm': delta_norm,
            'angle': angle,
            'cos_sim': cos_sim
        })
        if len(self.poincare_state['trajectory_buffer']) > 3:
            self.poincare_state['trajectory_buffer'].pop(0)

        # Restore original parameters before applying adjusted update
        for p, orig_p in zip(group['params'], self.poincare_state['original_params']):
            p.copy_(orig_p)

        # Poincaré-inspired update rule
        for p, g_orig, delta in zip(group['params'],
                                    self.poincare_state['original_grads'],
                                    delta_params):
            if delta_norm < eps:
                # Stability-preserving zero-displacement handler
                adjusted_grad = g_orig
            else:
                # Dynamic directional enhancement
                dir_projection = (g_orig.flatten() @ delta.flatten()) / delta_norm**2

                if angle < math.pi/4:  # Strong alignment
                    # Exponential convergence boost
                    adj_factor = c_scale * (1.0 + math.log(1.0 + 1.0/(delta_norm + eps)))
                    adjusted_grad = g_orig + adj_factor * delta/delta_norm
                elif angle > 3*math.pi/4:  # Strong opposition
                    # Damped oscillation correction
                    adjusted_grad = 0.5 * dir_projection * delta
                else:  # Neutral region
                    # Momentum-weighted adjustment
                    adjusted_grad = (1 - math.sin(angle)) * g_orig + \
                                    math.sin(angle) * dir_projection * delta

            # Update with stabilized momentum
            buf = self.state[p]['momentum_buffer']
            buf.mul_(momentum**self.poincare_state['convergence_factor']).add_(
                adjusted_grad, alpha=lr)
            p.sub_(buf)

        # Adaptive convergence factor based on trajectory stability
        if len(self.poincare_state['trajectory_buffer']) >= 2:
            recent_changes = [x['delta_norm'] for x in self.poincare_state['trajectory_buffer'][-2:]]
            self.poincare_state['convergence_factor'] = 1.0 / (1.0 + math.log1p(sum(recent_changes)))

        # Reset phase for next batch
        self.poincare_state['phase'] = 0
        return loss