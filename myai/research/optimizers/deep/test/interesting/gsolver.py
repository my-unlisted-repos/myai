# pylint:disable=signature-differs, not-callable # type:ignore
"""Two versions"""
from collections import deque

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer


def flatten_params(params):
    """Flatten a list of parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in params])

def unflatten_params(flat_params, params):
    """Unflatten a vector into the parameters of a model."""
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset+numel].view_as(p.data))
        offset += numel

class GSolver(Optimizer):
    """Its a zeroth-order method similar to BFGS but it estimates the gradient instead of the hessian inverse.
    But it is prone to getting stuck, in this version."""
    def __init__(self, params, lr=0.1, m_initial=10, m_history=100, epsilon=1e-3, kaczmarz_steps=10):
        defaults = dict(m_initial=m_initial, m_history=m_history, epsilon=epsilon, alpha=lr, kaczmarz_steps=kaczmarz_steps)
        super().__init__(params, defaults)
        self.state = self.param_groups[0]  # Using first param group's state for simplicity
        self.state.setdefault('step_count', 0)
        self.state.setdefault('initial_phase', True)
        self.state.setdefault('history', deque(maxlen=m_history))
        self.state.setdefault('L_old', None)
        self.state.setdefault('g_est', None)
        self.state.setdefault('initial_deltas', [])
        self.params = params

    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        with torch.no_grad():
            group = self.param_groups[0]
            m_initial = group['m_initial']
            m_history = group['m_history']
            epsilon = group['epsilon']
            alpha = group['alpha']
            kaczmarz_steps = group['kaczmarz_steps']
            params = group['params']

            state = self.state

            if state['step_count'] == 0:
                # Initialize parameters and initial loss
                flat_params = flatten_params(params)
                state['flat_params'] = flat_params.clone()
                state['L_old'] = closure(False)
                state['g_est'] = torch.zeros_like(flat_params)

            flat_params = state['flat_params']
            L_old = state['L_old']
            g_est = state['g_est']
            history = state['history']

            if state['initial_phase']:
                # initial gradient estimation on 1st step, via multiple random vectors, this needs to be fairly accurate for the rest to work well.
                for _ in range(m_initial):
                    delta = torch.randn_like(flat_params) * epsilon
                    # Save current parameters
                    original_flat = flat_params.clone()
                    # Perturb parameters
                    new_flat = original_flat + delta
                    unflatten_params(new_flat, params)
                    L_i = closure(False)
                    diff = L_i - L_old
                    state['initial_deltas'].append((delta, diff))
                    # Restore original parameters
                    unflatten_params(original_flat, params)
                # Compute initial gradient estimate using Kaczmarz on initial_deltas
                g_est.zero_()
                for _ in range(kaczmarz_steps):
                    for delta, diff in state['initial_deltas']:
                        delta = delta.to(g_est.device)
                        diff = diff.to(g_est.device)
                        residual = diff - torch.dot(delta, g_est)
                        denominator = torch.dot(delta, delta) + 1e-8
                        g_est.add_((residual / denominator) * delta)
                state['initial_phase'] = False
                # Take the first step with the initial gradient estimate
                new_flat = flat_params - alpha * g_est
                unflatten_params(new_flat, params)
                L_new = closure(False)
                diff_step = L_new - L_old
                delta_step = new_flat - flat_params
                history.append((delta_step, diff_step))
                state['flat_params'] = new_flat
                state['L_old'] = L_new
                state['step_count'] += 1
            else:
                # Update parameters using current gradient estimate
                new_flat = flat_params - alpha * g_est
                unflatten_params(new_flat, params)
                L_new = closure(False)
                diff_step = L_new - L_old
                delta_step = new_flat - flat_params
                history.append((delta_step, diff_step))
                # Update gradient estimate using Kaczmarz steps on history
                for _ in range(kaczmarz_steps):
                    for delta, diff in history:
                        residual = diff - torch.dot(delta, g_est)
                        denominator = torch.dot(delta, delta) + 1e-8
                        g_est.add_((residual / denominator) * delta)
                # Update state
                state['flat_params'] = new_flat
                state['L_old'] = L_new
                state['step_count'] += 1
        return L_new



class GSolverV2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, num_init=10, memory_size=10,
                 exploration_interval=20, exploration_scale=0.1,
                 eps=1e-8, max_grad_norm=1.0, accept_better_exploration=True):
        defaults = dict(lr=lr, num_init=num_init, memory_size=memory_size,
                        exploration_interval=exploration_interval,
                        exploration_scale=exploration_scale, eps=eps,
                        max_grad_norm=max_grad_norm,
                        accept_better_exploration=accept_better_exploration)
        super().__init__(params, defaults)

        self._state = {}

        for group in self.param_groups:
            group_id = id(group)
            self._state[group_id] = {
                'step': 0,
                'history': [],
                'prev_loss': None,
                'grad_est': None,
                'current_params': None,
                'num_params': 0,
                'best_loss': float('inf')
            }

            params = group['params']
            if any(p is not None for p in params):
                flat_params = parameters_to_vector(params)
                state = self._state[group_id]
                state['grad_est'] = torch.zeros_like(flat_params)
                state['current_params'] = flat_params.clone()
                state['num_params'] = len(flat_params)

    @torch.no_grad
    def step(self, closure):
        loss = None

        for group in self.param_groups:
            group_id = id(group)
            state = self._state[group_id]
            params = group['params']

            if not any(p is not None for p in params):
                continue

            lr = group['lr']
            num_init = group['num_init']
            memory_size = group['memory_size']
            exploration_interval = group['exploration_interval']
            exploration_scale = group['exploration_scale']
            eps = group['eps']
            max_grad_norm = group['max_grad_norm']
            accept_better = group['accept_better_exploration']

            current_params = state['current_params']
            grad_est = state['grad_est']
            step = state['step']
            history = state['history']

            vector_to_parameters(current_params, params)

            if step < num_init:
                v = torch.randn_like(current_params) * exploration_scale
                new_params = current_params + v

                vector_to_parameters(new_params, params)
                loss_plus = closure(False)

                vector_to_parameters(current_params, params)
                loss_current = closure(False) if state['prev_loss'] is None else state['prev_loss']

                delta_f = loss_plus - loss_current
                v_norm_sq = torch.dot(v, v) + eps
                grad_update = (delta_f / v_norm_sq) * v

                grad_update_norm = torch.norm(grad_update)
                if grad_update_norm > max_grad_norm:
                    grad_update.mul_(max_grad_norm / (grad_update_norm + eps))

                if step == 0:
                    grad_est.copy_(grad_update)
                else:
                    grad_est.mul_(step/(step+1)).add_(grad_update/(step+1))

                state['prev_loss'] = loss_plus
                state['best_loss'] = min(state['best_loss'], loss_plus.item())
                loss = loss_plus
            else:
                with torch.no_grad():
                    grad_norm = torch.norm(grad_est)
                    if grad_norm > max_grad_norm:
                        grad_est.mul_(max_grad_norm / (grad_norm + eps))

                    new_params = current_params - lr * grad_est
                    vector_to_parameters(new_params, params)
                    loss_new = closure(False)

                    if loss_new.item() < state['prev_loss'] or step % exploration_interval == 0:
                        # Accept step
                        delta_f_new = loss_new - state['prev_loss']
                        s = new_params - current_params
                        history.append((s.clone(), delta_f_new.item()))
                        current_params.copy_(new_params)
                        state['prev_loss'] = loss_new
                        state['best_loss'] = min(state['best_loss'], loss_new.item())
                        loss = loss_new
                    else:
                        # Reject step and reduce learning rate
                        vector_to_parameters(current_params, params)
                        lr *= 0.8
                        loss = state['prev_loss']

                    if len(history) > memory_size:
                        history.pop(0)

                    valid_history = [(s, df) for s, df in history
                                   if torch.isfinite(torch.tensor(df))
                                   and torch.norm(s) > eps]

                    q = grad_est.clone()
                    alpha_list = []

                    for s_i, delta_f_i in reversed(valid_history):
                        s_i = s_i.to(q.device)
                        s_norm = torch.norm(s_i)
                        if s_norm < eps:
                            continue

                        rho_i = 1.0 / (s_norm**2 + eps)
                        alpha_i = rho_i * (delta_f_i - torch.dot(s_i, q))
                        alpha_list.append(alpha_i.item())
                        q.add_(alpha_i * s_i, alpha=-1)

                    z = q.clone()
                    for (s_i, delta_f_i), alpha_i in zip(valid_history, reversed(alpha_list)):
                        s_i = s_i.to(z.device)
                        s_norm = torch.norm(s_i)
                        if s_norm < eps:
                            continue

                        rho_i = 1.0 / (s_norm**2 + eps)
                        beta_i = rho_i * torch.dot(s_i, z)
                        z.add_((alpha_i - beta_i) * s_i)

                    # update gradient estimate
                    update_norm = torch.norm(z)
                    if update_norm > max_grad_norm:
                        z.mul_(max_grad_norm / (update_norm + eps))

                    grad_est.mul_(0.9).add_(z, alpha=0.1)

                    # exploration step so that it doesnt get stuck
                    if step % exploration_interval == 0:
                        # Generate orthogonal direction with clamped magnitude
                        rand_dir = torch.randn_like(current_params)
                        grad_dir = grad_est / (torch.norm(grad_est) + eps)

                        # Orthogonalize and normalize
                        rand_dir -= torch.dot(rand_dir, grad_dir) * grad_dir
                        rand_norm = torch.norm(rand_dir)
                        if rand_norm > eps:
                            rand_dir /= rand_norm

                        # exploration scale based on recent progress
                        progress = state['best_loss'] - state['prev_loss'].item()
                        explore_scale = exploration_scale * max(0.1, 1 - abs(progress))
                        v = explore_scale * rand_dir

                        # Evaluate exploration direction
                        new_params_explore = current_params + v
                        vector_to_parameters(new_params_explore, params)
                        loss_explore = closure(False)

                        # Accept exploration if better and allowed
                        if accept_better and loss_explore.item() < state['prev_loss'].item():
                            current_params.copy_(new_params_explore)
                            state['prev_loss'] = loss_explore
                            state['best_loss'] = min(state['best_loss'], loss_explore.item())
                            loss = loss_explore
                            # Reset history to avoid inconsistent state
                            history.clear()
                        else:
                            vector_to_parameters(current_params, params)

                        # Update history with clamped delta_f
                        delta_f_explore = max(min(loss_explore.item() - state['prev_loss'].item(), 1e3), -1e3)
                        history.append((v.clone(), delta_f_explore))
                        if len(history) > memory_size:
                            history.pop(0)

            state['step'] += 1

        return loss