import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import math
import matplotlib.pyplot as plt
import numpy as np

class SepAdam(Optimizer):
    """Implements Separated Adam (SepAdam) optimizer."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta_d=0.99,
                 eps=1e-8, eps_d=1e-12, weight_decay=0, fallback_adam_steps=10):
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients for Adam components (default: (0.9, 0.999))
            beta_d (float, optional): coefficient for direction EMA (default: 0.99)
            eps (float, optional): term added to denominators for stability (default: 1e-8)
            eps_d (float, optional): term added for direction norm stability (default: 1e-12)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            fallback_adam_steps (int, optional): Number of initial steps to use standard Adam
                                                 before activating directional logic (default: 10)
        """
        if not 0.0 <= lr: raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= eps_d: raise ValueError(f"Invalid eps_d: {eps_d}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= beta_d < 1.0: raise ValueError(f"Invalid beta_d: {beta_d}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not fallback_adam_steps >= 0: raise ValueError(f"Invalid fallback_adam_steps: {fallback_adam_steps}")

        defaults = dict(lr=lr, betas=betas, beta_d=beta_d, eps=eps, eps_d=eps_d,
                        weight_decay=weight_decay, fallback_adam_steps=fallback_adam_steps)
        super(SepAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SepAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta_d = group['beta_d']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            eps_d = group['eps_d']
            fallback_steps = group['fallback_adam_steps']

            all_params_list = []
            all_grads_list = []
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('SepAdam does not support sparse gradients')
                all_params_list.append(p)
                # Apply weight decay here before flattening
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                all_grads_list.append(grad)

            if not all_params_list:
                continue

            # Use collective state for direction finding and separated states
            state_collective = self.state.get('__collective_state__')
            if state_collective is None:
                state_collective = {}
                state_collective['step'] = 0
                # Standard Adam state (for fallback and initialization)
                flat_param_example = torch.cat([p.data.view(-1) for p in all_params_list])
                state_collective['exp_avg_adam'] = torch.zeros_like(flat_param_example)
                state_collective['exp_avg_sq_adam'] = torch.zeros_like(flat_param_example)
                # Direction state
                state_collective['exp_avg_delta'] = torch.zeros_like(flat_param_example)
                state_collective['prev_flat_params'] = flat_param_example.clone()
                # Separated states
                state_collective['exp_avg_par'] = torch.tensor(0.0, device=flat_param_example.device) # Scalar
                state_collective['exp_avg_sq_par'] = torch.tensor(0.0, device=flat_param_example.device) # Scalar
                state_collective['exp_avg_perp'] = torch.zeros_like(flat_param_example) # Vector
                state_collective['exp_avg_sq_perp'] = torch.zeros_like(flat_param_example) # Vector
                self.state['__collective_state__'] = state_collective

            state_collective['step'] += 1
            step = state_collective['step']

            # Flatten current params and grads
            flat_params = torch.cat([p.data.view(-1) for p in all_params_list])
            flat_grads = torch.cat([g.view(-1) for g in all_grads_list]) # Grads already have weight decay

            # === Get Adam step (used for fallback or comparison) ===
            exp_avg_adam = state_collective['exp_avg_adam']
            exp_avg_sq_adam = state_collective['exp_avg_sq_adam']
            exp_avg_adam.mul_(beta1).add_(flat_grads, alpha=1 - beta1)
            exp_avg_sq_adam.mul_(beta2).addcmul_(flat_grads, flat_grads.conj(), value=1 - beta2) # Use conj() for complex support? Assume real.

            bc1 = 1 - beta1 ** step
            bc2 = 1 - beta2 ** step

            m_hat_adam = exp_avg_adam / bc1
            v_hat_adam = exp_avg_sq_adam / bc2
            denom_adam = v_hat_adam.sqrt().add_(eps)
            u_adam = m_hat_adam / denom_adam

            # === Direction Estimation ===
            v_dir = None
            direction_reliable = False
            if step > 1:
                 delta_x_prev = flat_params - state_collective['prev_flat_params']
                 exp_avg_delta = state_collective['exp_avg_delta']
                 exp_avg_delta.mul_(beta_d).add_(delta_x_prev, alpha=1 - beta_d)

                 # No bias correction for delta needed usually, let it warm up
                 d_hat = exp_avg_delta
                 d_norm = torch.linalg.norm(d_hat)

                 if d_norm > eps_d and step > fallback_steps: # Only use direction after fallback period
                     v_dir = d_hat / d_norm
                     direction_reliable = True

            # Store current flat params for next iter
            state_collective['prev_flat_params'] = flat_params.clone()

            # === Calculate Update Step ===
            if direction_reliable:
                try:
                    # Project gradient
                    g_parallel_comp = torch.dot(flat_grads, v_dir)
                    g_parallel = g_parallel_comp * v_dir
                    g_perp = flat_grads - g_parallel

                    # Update parallel state (scalar)
                    exp_avg_par = state_collective['exp_avg_par']
                    exp_avg_sq_par = state_collective['exp_avg_sq_par']
                    exp_avg_par.mul_(beta1).add_(g_parallel_comp, alpha=1 - beta1)
                    exp_avg_sq_par.mul_(beta2).add_(g_parallel_comp**2, alpha=1 - beta2)

                    m_hat_par = exp_avg_par / bc1
                    v_hat_par = exp_avg_sq_par / bc2
                    denom_par = torch.sqrt(v_hat_par).add_(eps)
                    update_parallel_scalar = m_hat_par / denom_par

                    # Update perpendicular state (vector)
                    exp_avg_perp = state_collective['exp_avg_perp']
                    exp_avg_sq_perp = state_collective['exp_avg_sq_perp']
                    exp_avg_perp.mul_(beta1).add_(g_perp, alpha=1 - beta1)
                    exp_avg_sq_perp.mul_(beta2).addcmul_(g_perp, g_perp, value=1 - beta2)

                    m_hat_perp = exp_avg_perp / bc1
                    v_hat_perp = exp_avg_sq_perp / bc2
                    denom_perp = v_hat_perp.sqrt().add_(eps)
                    update_perp = m_hat_perp / denom_perp

                    # Combine
                    u_final_flat = update_parallel_scalar * v_dir + update_perp

                    # Sanity check for NaNs/Infs in update components
                    if not torch.isfinite(update_parallel_scalar) or not torch.all(torch.isfinite(update_perp)):
                         print(f"Warning: Step {step}, NaN/Inf detected in SepAdam components. Falling back to Adam step.")
                         # print(f"  m_hat_par={m_hat_par.item():.2e}, v_hat_par={v_hat_par.item():.2e}, denom_par={denom_par.item():.2e}")
                         # print(f"  ||m_hat_perp||={torch.linalg.norm(m_hat_perp):.2e}, ||v_hat_perp||={torch.linalg.norm(v_hat_perp):.2e}")
                         u_final_flat = u_adam # Fallback to standard Adam step if calculation blows up

                except Exception as e:
                    print(f"Error during SepAdam calculation at step {step}: {e}. Falling back to Adam.")
                    u_final_flat = u_adam
            else:
                # Use standard Adam during fallback period or if direction is unreliable
                u_final_flat = u_adam


            # === Apply Update ===
            start_idx = 0
            for p in all_params_list:
                num_elm = p.numel()
                p.add_(u_final_flat[start_idx : start_idx + num_elm].view_as(p), alpha=-lr)
                start_idx += num_elm

        return loss

