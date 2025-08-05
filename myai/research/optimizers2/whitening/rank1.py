import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer


class GraND(Optimizer):
    """Implements Gradient Rank-1 Normalized Descent optimizer."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta_u=0.99,
                 C=1.0, eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, beta_u=beta_u, C=C, eps=eps,
                        weight_decay=weight_decay)
        super(GraND, self).__init__(params, defaults)

        # Initialize scalar states v_u and v_g (shared across all param groups for now)
        # In a more complex scenario, these might be per parameter group or device
        self.state['shared_v_u'] = torch.tensor(0.0)
        self.state['shared_v_g'] = torch.tensor(0.0)
        self.state['shared_step'] = 0 # Shared step counter for bias correction

    def __setstate__(self, state):
        super(GraND, self).__setstate__(state)
        # Ensure shared states are correctly handled if optimizer is loaded/saved
        if 'shared_v_u' not in self.state:
             self.state['shared_v_u'] = torch.tensor(0.0)
        if 'shared_v_g' not in self.state:
             self.state['shared_v_g'] = torch.tensor(0.0)
        if 'shared_step' not in self.state:
             self.state['shared_step'] = 0

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Use a shared step across all parameter groups for scalar state bias correction
        self.state['shared_step'] += 1
        shared_step = self.state['shared_step']
        beta_v = self.defaults['betas'][1] # Use beta2 for variance estimates

        # Aggregate preconditioned gradient info across all parameters
        g_prec_list = []
        m_prec_list = []
        d_inv_list = []
        param_list = []
        num_params_total = 0

        # --- First loop: Calculate Adam stats and preconditioned grads/momentum ---
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('GraND does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of normalized preconditioned momentum direction
                    state['u_dir'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq, u_dir = state['exp_avg'], state['exp_avg_sq'], state['u_dir']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Adam bias correction
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Preconditioning
                d_inv = v_hat.sqrt().add_(eps).pow_(-1) # 1 / (sqrt(v) + eps)
                g_prec = d_inv * grad
                m_prec = d_inv * m_hat

                g_prec_list.append(g_prec.flatten())
                m_prec_list.append(m_prec.flatten())
                d_inv_list.append(d_inv.flatten())
                param_list.append(p)
                num_params_total += p.numel()


        if num_params_total == 0:
             return loss # No parameters with grads

        # Flatten into single tensors for global calculations
        g_prec_flat = torch.cat(g_prec_list)
        m_prec_flat = torch.cat(m_prec_list)

        # --- Update shared states (u, v_u, v_g) ---
        beta_u = self.defaults['beta_u']
        eps = self.defaults['eps']
        C = self.defaults['C']

        # Get shared state device (assume all params on same device for simplicity)
        device = param_list[0].device
        shared_v_u = self.state['shared_v_u'].to(device)
        shared_v_g = self.state['shared_v_g'].to(device)

        # Initialize or get global 'u' direction (must match flat tensor size)
        if 'global_u' not in self.state or self.state['global_u'].numel() != num_params_total:
             self.state['global_u'] = torch.randn(num_params_total, device=device) # Random init
             self.state['global_u'] /= (torch.norm(self.state['global_u']) + eps)
        global_u = self.state['global_u']

        # 5. Update dominant direction u (using m_prec for stability)
        m_prec_norm = torch.norm(m_prec_flat)
        if m_prec_norm > eps:
            dir_t = m_prec_flat / m_prec_norm
            global_u.mul_(beta_u).add_(dir_t, alpha=1 - beta_u)
            global_u.div_(torch.norm(global_u) + eps) # Re-normalize
        # Store updated global_u back into state
        self.state['global_u'] = global_u

        # 6. Update directional and total variances (using g_prec)
        g_prec_norm_sq = torch.dot(g_prec_flat, g_prec_flat)
        # Ensure global_u is normalized before dot product
        proj_g_u = torch.dot(g_prec_flat, global_u)

        shared_v_u = beta_v * shared_v_u + (1 - beta_v) * proj_g_u**2
        shared_v_g = beta_v * shared_v_g + (1 - beta_v) * g_prec_norm_sq

        # Store updated shared states
        self.state['shared_v_u'] = shared_v_u
        self.state['shared_v_g'] = shared_v_g

        # 7. Bias correction for shared variances (use shared step)
        bias_correction_v_shared = 1 - beta_v ** shared_step
        v_u_hat = shared_v_u / bias_correction_v_shared
        v_g_hat = shared_v_g / bias_correction_v_shared

        # 8. Calculate directional damping
        avg_var_per_dim = v_g_hat / (num_params_total + eps)
        # Prevent division by zero or negative sqrt if avg_var_per_dim is tiny/zero
        sqrt_avg_var = torch.sqrt(F.relu(avg_var_per_dim) + eps**2)
        sqrt_v_u = torch.sqrt(F.relu(v_u_hat) + eps**2) # Use relu for safety

        damping_ratio = sqrt_v_u / (sqrt_avg_var + eps)
        # C controls sensitivity; only damp if ratio > 1
        damping_factor = 1.0 + C * F.relu(damping_ratio - 1.0)
        # Prevent extreme damping
        damping_factor = torch.clamp(damping_factor, min=1.0, max=1.0 / eps)


        # 9. Project preconditioned momentum and apply damping
        m_parallel_scalar = torch.dot(m_prec_flat, global_u)
        m_parallel = m_parallel_scalar * global_u
        m_perp = m_prec_flat - m_parallel
        m_adj_flat = m_parallel / (damping_factor + eps) + m_perp # Damp parallel component

        # --- Second loop: Apply update to parameters ---
        # Split the adjusted momentum and D_inv back to original parameter shapes
        m_adj_split = torch.split(m_adj_flat, [p.numel() for p in param_list])
        d_inv_split = torch.split(torch.cat(d_inv_list), [p.numel() for p in param_list])

        for i, p in enumerate(param_list):
             group = self.param_groups[0] # Assume one group for simplicity here
             lr = group['lr']

             m_adj = m_adj_split[i].view_as(p)
             d_inv = d_inv_split[i].view_as(p)

             # 10. Compute final update step (re-applying D_inv)
             # delta_w = -lr * D_inv * m_adj # Error here, D_inv was already applied to m_prec
             # Need to apply D_inv back to the *adjusted* preconditioned momentum
             delta_w = -lr * (d_inv * m_adj)

             # 11. Update parameters
             p.add_(delta_w)

        return loss


class GraNDScaleFree(Optimizer):
    """
    Implements Gradient Rank-1 Normalized Descent optimizer with adaptive
    step size inspired by Adafactor, aiming to be LR tuning-free.
    """

    def __init__(self, params, lr=0.1, betas=(0.9, 0.999),
                 beta_u=0.99, beta_rms=0.999, C=1.0, eps_scale=1e-30,
                 eps_adam=1e-8, eps_rms=1e-8, weight_decay=0, use_sqrt_decay=True):

        defaults = dict(lr=lr, betas=betas,
                        beta_u=beta_u, beta_rms=beta_rms, C=C,
                        eps_scale=eps_scale, eps_adam=eps_adam, eps_rms=eps_rms,
                        weight_decay=weight_decay, use_sqrt_decay=use_sqrt_decay)
        super().__init__(params, defaults)

        # Initialize shared scalar states
        self.state['shared_v_u'] = torch.tensor(0.0)
        self.state['shared_v_g'] = torch.tensor(0.0)
        self.state['shared_param_rms_ema'] = torch.tensor(0.0) # EMA of parameter RMS
        self.state['shared_step'] = 0 # Shared step counter

    def __setstate__(self, state):
        super(GraNDScaleFree, self).__setstate__(state)
        # Ensure shared states are correctly handled
        if 'shared_v_u' not in self.state:
             self.state['shared_v_u'] = torch.tensor(0.0)
        if 'shared_v_g' not in self.state:
             self.state['shared_v_g'] = torch.tensor(0.0)
        if 'shared_param_rms_ema' not in self.state:
             self.state['shared_param_rms_ema'] = torch.tensor(0.0)
        if 'shared_step' not in self.state:
             self.state['shared_step'] = 0

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.state['shared_step'] += 1
        shared_step = self.state['shared_step']
        beta_v = self.defaults['betas'][1]
        beta_rms = self.defaults['beta_rms']
        eps_adam = self.defaults['eps_adam']
        eps_rms = self.defaults['eps_rms']
        eps_scale = self.defaults['eps_scale']

        # --- Aggregation lists ---
        g_prec_list = []
        m_prec_list = []
        d_inv_list = []
        param_list = []
        param_sq_sum_list = []
        num_params_total = 0

        # --- First loop: Calculate Adam stats and preconditioned grads/momentum ---
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('GraNDScaleFree does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['u_dir'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Keep track of per-parameter step for bias correction
                state['step'] += 1

                # Perform stepweight decay
                if weight_decay != 0:
                     grad = grad.add(p, alpha=weight_decay)

                # Adam moment updates
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2) # Use conj() for complex support

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Preconditioning
                d_inv = v_hat.sqrt().add_(eps_adam).pow_(-1)
                g_prec = d_inv * grad
                m_prec = d_inv * m_hat

                g_prec_list.append(g_prec.flatten())
                m_prec_list.append(m_prec.flatten())
                d_inv_list.append(d_inv.flatten())
                param_list.append(p)
                param_sq_sum_list.append(torch.sum(p**2))
                num_params_total += p.numel()

        if num_params_total == 0:
             return loss

        # --- Global calculations (device placement assumed consistent) ---
        device = param_list[0].device
        g_prec_flat = torch.cat(g_prec_list)
        m_prec_flat = torch.cat(m_prec_list)

        # --- Update shared GraND states (u, v_u, v_g) ---
        beta_u = self.defaults['beta_u']
        C = self.defaults['C']
        shared_v_u = self.state['shared_v_u'].to(device)
        shared_v_g = self.state['shared_v_g'].to(device)

        if 'global_u' not in self.state or self.state['global_u'].numel() != num_params_total:
             self.state['global_u'] = torch.randn(num_params_total, device=device)
             self.state['global_u'] /= (torch.norm(self.state['global_u']) + eps_rms)
        global_u = self.state['global_u']

        # Update dominant direction u
        m_prec_norm = torch.norm(m_prec_flat)
        if m_prec_norm > eps_rms:
            dir_t = m_prec_flat / m_prec_norm
            global_u.mul_(beta_u).add_(dir_t, alpha=1 - beta_u)
            global_u.div_(torch.norm(global_u) + eps_rms)
        self.state['global_u'] = global_u

        # Update directional and total variances
        g_prec_norm_sq = torch.dot(g_prec_flat, g_prec_flat)
        proj_g_u = torch.dot(g_prec_flat, global_u)
        shared_v_u = beta_v * shared_v_u + (1 - beta_v) * proj_g_u**2
        shared_v_g = beta_v * shared_v_g + (1 - beta_v) * g_prec_norm_sq
        self.state['shared_v_u'] = shared_v_u
        self.state['shared_v_g'] = shared_v_g

        # Bias correction for shared variances
        bias_correction_v_shared = 1 - beta_v ** shared_step
        v_u_hat = shared_v_u / bias_correction_v_shared
        v_g_hat = shared_v_g / bias_correction_v_shared

        # Calculate directional damping factor
        avg_var_per_dim = v_g_hat / (num_params_total + eps_rms)
        sqrt_avg_var = torch.sqrt(F.relu(avg_var_per_dim) + eps_rms**2)
        sqrt_v_u = torch.sqrt(F.relu(v_u_hat) + eps_rms**2)
        damping_ratio = sqrt_v_u / (sqrt_avg_var + eps_rms)
        damping_factor = 1.0 + C * F.relu(damping_ratio - 1.0)
        damping_factor = torch.clamp(damping_factor, min=1.0, max=1.0 / eps_rms)

        # Project preconditioned momentum and apply damping
        m_parallel_scalar = torch.dot(m_prec_flat, global_u)
        m_parallel = m_parallel_scalar * global_u
        m_perp = m_prec_flat - m_parallel
        m_adj_flat = m_parallel / (damping_factor + eps_rms) + m_perp

        # --- Calculate Raw Update and RMS ---
        # Split necessary components back to parameter shapes
        m_adj_split = torch.split(m_adj_flat, [p.numel() for p in param_list])
        d_inv_split = torch.split(torch.cat(d_inv_list), [p.numel() for p in param_list])

        raw_update_list = []
        for i, p in enumerate(param_list):
            m_adj = m_adj_split[i].view_as(p)
            d_inv = d_inv_split[i].view_as(p)
            raw_update_p = d_inv * m_adj # This is the direction GraND suggests
            raw_update_list.append(raw_update_p.flatten())

        raw_update_flat = torch.cat(raw_update_list)
        rms_raw_update = torch.sqrt(torch.mean(raw_update_flat**2))

        # --- Calculate Adaptive Step Size ---
        # Update EMA of parameter RMS
        shared_param_rms_ema = self.state['shared_param_rms_ema'].to(device)
        current_param_sq_sum = sum(sq for sq in param_sq_sum_list) # Use .item() to avoid graph
        current_rms_param = torch.sqrt(current_param_sq_sum / num_params_total)
        shared_param_rms_ema = beta_rms * shared_param_rms_ema + (1 - beta_rms) * current_rms_param
        self.state['shared_param_rms_ema'] = shared_param_rms_ema # Store back

        # Compute step schedule
        step_schedule = 1.0
        if self.defaults['use_sqrt_decay']:
             # Start decay after a few steps? Or immediately? Let's start immediately.
             # Adafactor uses max(1e-2, 1/sqrt(t)) sometimes. Add small floor.
             step_schedule = max(1e-3, 1.0 / math.sqrt(shared_step))

        # Compute adaptive learning rate / magnitude
        lr = self.param_groups[0]['lr']
        adaptive_magnitude = (
            lr *
            max(shared_param_rms_ema.item(), eps_scale) * # Use .item()
            step_schedule
        )

        # --- Final Update Calculation ---
        # Scale the normalized raw update
        # Factor to scale the raw_update so its RMS becomes adaptive_magnitude
        scaling_factor = adaptive_magnitude / (rms_raw_update + eps_rms)

        # Prevent excessively large steps if rms_raw_update is tiny
        scaling_factor = min(scaling_factor, lr * 1e4) # Heuristic cap

        final_update_flat = -scaling_factor * raw_update_flat

        # --- Second loop: Apply update to parameters ---
        final_update_split = torch.split(final_update_flat, [p.numel() for p in param_list])
        for i, p in enumerate(param_list):
             delta_w = final_update_split[i].view_as(p)
             p.add_(delta_w)

        return loss