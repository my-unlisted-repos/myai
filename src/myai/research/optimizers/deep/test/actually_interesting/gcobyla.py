# pylint:disable=signature-differs, not-callable

import math

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer


class GCOBYLA(Optimizer):
    """COBYLA (sort of COBYQA) with gradient information. Works well."""
    def __init__(self, params, delta=0.1, max_delta=1.0, min_delta=1e-9, eta=0.001,
                 history_size=5, max_consecutive_rejections=5, noise_scale=1e-8,
                 curvature_decay=0.9, direction_memory=0.5):
        defaults = dict(delta=delta, max_delta=max_delta, min_delta=min_delta, eta=eta,
                        history_size=history_size, max_consecutive_rejections=max_consecutive_rejections,
                        noise_scale=noise_scale, curvature_decay=curvature_decay,
                        direction_memory=direction_memory)
        super().__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('consecutive_rejections', 0)
            group.setdefault('prev_loss', math.inf)
            group.setdefault('curvature_trust', 0.0)
            group.setdefault('prev_direction', None)
            group.setdefault('oscillation_counter', 0)

    def step(self, closure):
        closure = torch.enable_grad()(closure)
        final_loss = None

        for group in self.param_groups:
            params = [p for p in group['params'] if p.requires_grad]
            if not params:
                continue

            # ---- Initial evaluation ----
            current_loss = closure()
            current_loss_value = current_loss.item()
            flat_params = parameters_to_vector(params)
            flat_grad = parameters_to_vector([p.grad for p in params])

            # ---- Adaptive noise injection ----
            param_scale = torch.norm(flat_params).item()
            effective_noise = group['noise_scale'] * min(1.0, group['delta']/group['max_delta'])
            min_grad_norm = max(1e-12, effective_noise * 0.1)

            g_norm = torch.norm(flat_grad).item()
            if g_norm < min_grad_norm and effective_noise > 0:
                noise = torch.randn_like(flat_grad)
                flat_grad = noise / torch.norm(noise) * effective_noise
                g_norm = effective_noise

            # ---- Direction computation ----
            state = self.state[params[0]]
            step_history = state.get('step_history', [])
            grad_history = state.get('grad_history', [])

            # Base gradient direction
            grad_dir = -flat_grad / g_norm
            final_dir = grad_dir.clone()

            # Curvature-aware adjustment with decay
            if len(step_history) > 0 and group['curvature_trust'] > 0.1:
                prev_step = step_history[-1]
                prev_grad = grad_history[-1]
                delta_grad = flat_grad - prev_grad

                # Secant equation adjustment
                sy = torch.dot(prev_step, delta_grad).item()
                if abs(sy) > 1e-12:
                    curvature_adj = torch.dot(prev_step, prev_step).item() / sy
                    final_dir += group['curvature_trust'] * curvature_adj * grad_dir

            # Direction smoothing with memory
            if 'prev_direction' in state and state['prev_direction'] is not None:
                final_dir = (1 - group['direction_memory']) * final_dir + \
                            group['direction_memory'] * state['prev_direction']

            # Normalize direction
            dir_norm = torch.norm(final_dir).item()
            if dir_norm > 1e-12:
                final_dir /= dir_norm
            else:
                final_dir = grad_dir

            # ---- Oscillation detection ----
            if 'prev_direction' in state and state['prev_direction'] is not None:
                direction_change = 1 - torch.dot(final_dir, state['prev_direction']).item()
                if direction_change > 0.5:  # >60 degree change
                    group['oscillation_counter'] += 1
                else:
                    group['oscillation_counter'] = max(0, group['oscillation_counter'] - 1)

            # ---- Trust-region step ----
            step = final_dir * group['delta']
            orig_params = flat_params.clone().detach()
            orig_grads = [p.grad.clone() for p in params]

            # ---- Tentative evaluation ----
            vector_to_parameters(orig_params + step, params)
            new_loss = closure()
            new_loss_value = new_loss.item()

            # ---- Smart rho calculation ----
            actual_reduction = group['prev_loss'] - new_loss_value
            predicted_reduction = -torch.dot(flat_grad, step).item()
            rho = actual_reduction / predicted_reduction if abs(predicted_reduction) > 1e-12 else 0.0
            group['prev_loss'] = new_loss_value

            # ---- Trust region adaptation ----
            consecutive_rejections = group['consecutive_rejections']

            if rho > group['eta'] or (actual_reduction > 0 and abs(rho) < 0.1):
                # Accept step
                group['consecutive_rejections'] = 0
                state['step_history'] = (step_history + [step.detach()])[-group['history_size']:]
                state['grad_history'] = (grad_history + [flat_grad.detach()])[-group['history_size']:]
                state['prev_direction'] = final_dir.detach()
                final_loss = new_loss_value

                # Update curvature trust factor
                if len(step_history) >= 2:
                    group['curvature_trust'] = min(1.0, group['curvature_trust'] * 1.1)

                # Oscillation-aware expansion
                if group['oscillation_counter'] < 2:
                    if rho > 0.75:
                        group['delta'] = min(group['delta'] * 1.5, group['max_delta'])
                    elif rho > 0.25:
                        group['delta'] = min(group['delta'] * 1.2, group['max_delta'])
            else:
                # Reject step
                vector_to_parameters(orig_params, params)
                for p, og in zip(params, orig_grads):
                    p.grad.copy_(og)
                final_loss = current_loss_value
                consecutive_rejections += 1
                group['consecutive_rejections'] = consecutive_rejections
                group['curvature_trust'] *= group['curvature_decay']

                # Smart contraction
                contraction = 0.5
                if consecutive_rejections > 1:
                    contraction = 0.25 if rho < 0 else 0.5
                if group['oscillation_counter'] > 2:
                    contraction *= 0.8

                new_delta = max(group['delta'] * contraction, group['min_delta'])

                # History reset if stuck
                if consecutive_rejections >= group['max_consecutive_rejections']:
                    state['step_history'] = []
                    state['grad_history'] = []
                    state['prev_direction'] = None
                    group['consecutive_rejections'] = 0
                    group['curvature_trust'] = 0.0
                    new_delta = min(group['delta'] * 2, group['max_delta'])

                group['delta'] = new_delta

        return final_loss