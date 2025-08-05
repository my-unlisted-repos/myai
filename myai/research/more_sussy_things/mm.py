import torch

class MatrixMomentumOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, mu, eps=1e-4):

        defaults = dict(lr=lr, mu=mu, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure):
        params = []
        prev_dws = []
        original_params = []

        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad: continue
                params.append(p)
                original_params.append(p.clone())
                state = self.state[p]
                if 'dw_prev' not in state:
                    state['dw_prev'] = torch.zeros_like(p)
                prev_dws.append(state['dw_prev'])

        hvps = [torch.zeros_like(p) for p in params]

        eps = self.param_groups[0]['eps']

        # Calculate ∇L(w + ε*Δw_prev)
        for p, dw_prev, orig in zip(params, prev_dws, original_params):
            if p.requires_grad:
                p.copy_(orig).add_(dw_prev, alpha=eps)

        with torch.enable_grad(): closure() # closure zeros grads, runs backward and sets new grads
        grads_plus = []
        for p in params:
            if p.grad is not None:
                grads_plus.append(p.grad.clone())
            else:
                grads_plus.append(torch.zeros_like(p))


        for p, dw_prev, orig in zip(params, prev_dws, original_params):
            if p.requires_grad:
                p.copy_(orig).add_(dw_prev, alpha=-eps)

        with torch.enable_grad(): closure()
        grads_minus = []
        for p in params:
            if p.requires_grad and p.grad is not None:
                grads_minus.append(p.grad.clone())
            else:
                grads_minus.append(torch.zeros_like(p))

        for p, orig in zip(params, original_params):
            if p.requires_grad:
                p.copy_(orig)

        # Calculate HVP components: (grad_plus - grad_minus) / (2 * epsilon)
        hvps = [(gp - gm).div_(2 * eps) for gp, gm in zip(grads_plus, grads_minus)]

        with torch.enable_grad(): loss = closure() # Closure zeros grads, runs model, returns loss

        param_idx_counter = 0
        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']

            for p in group['params']:
                if p.grad is None:
                    if p in params:
                         param_idx_counter +=1
                    continue

                state = self.state[p]
                if 'step' not in state: state['step'] = 0
                state['step'] += 1

                grad = p.grad

                prev_dw = prev_dws[param_idx_counter]
                hvp = hvps[param_idx_counter]

                grad_contribution = grad.mul(-lr)
                momentum_contribution = prev_dw.add(hvp, alpha=-mu)

                update = grad_contribution.add(momentum_contribution)

                p.add_(update)

                state['dw_prev'] = update.clone()

                param_idx_counter += 1

        return loss


class AutoMatrixMomentumOptimizer(torch.optim.Optimizer): # proposed by *experts*
    def __init__(self, params, lr, eps_div_guard=1e-8):
        """
        Implements Matrix Momentum with automatic mu estimation using forward finite differences for HVP.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): learning rate.
            eps_div_guard (float, optional): Small constant for numerical stability in mu_DB estimation,
                                             added to the denominator (default: 1e-8).
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps_div_guard:
            raise ValueError(f"Invalid eps_div_guard: {eps_div_guard}")

        defaults = dict(lr=lr, eps_div_guard=eps_div_guard)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure = None):
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        # --- Collect current gradients and load state from previous step ---
        params_with_grad = []
        current_grads_list = [] # List of g_k for each parameter (∇L(w_k))
        prev_dws_list = []      # List of s_k = Δw_{k-1} for each parameter (previous update)
        # List of g_{k-1} for each parameter (∇L(w_k - Δw_{k-1}) which is ∇L(w_{k-1}))
        prev_step_grads_list = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                params_with_grad.append(p)
                # Clone current gradient g_k
                current_grads_list.append(p.grad.clone(memory_format=torch.preserve_format))

                state = self.state[p]
                # State initialization for the first step
                if not state: # Check if state is empty (first step for this param)
                    state['step'] = 0
                    # s_0 = Δw_{-1} = 0 (no previous update)
                    state['dw_prev'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # g_{-1} = ∇L(w_0 - s_0) should be zero for y_0 = g_0 - 0 = g_0
                    state['grad_prev_step'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                prev_dws_list.append(state['dw_prev'])          # s_k = Δw_{k-1}
                prev_step_grads_list.append(state['grad_prev_step']) # g_{k-1}

        if not params_with_grad:
            return loss # No parameters with gradients, nothing to optimize

        # --- Calculate y_k = g_k - g_{k-1} (HVP approximation H_k * s_k) ---
        # y_k_list stores H_k * s_k for each parameter
        y_k_list = []
        for cg, pg_prev_step in zip(current_grads_list, prev_step_grads_list):
            y_k_list.append(cg.sub(pg_prev_step)) # y_k = g_k - g_{k-1}

        # --- Estimate mu_DB = ||s_k|| / ||y_k|| (scalar, using global norms) ---
        # s_k_global_flat: flattened vector of all prev_dws_list elements
        # y_k_global_flat: flattened vector of all y_k_list elements
        mu_DB = 0.0
        # Ensure lists are not empty before attempting to cat; params_with_grad check handles this.
        # Handle scalar parameters (numel=0 check is too strict, reshape(-1) handles scalars)
        s_k_global_flat = torch.cat([dw.reshape(-1) for dw in prev_dws_list], dim=0)
        y_k_global_flat = torch.cat([yk.reshape(-1) for yk in y_k_list], dim=0)

        norm_s_k_global = torch.norm(s_k_global_flat)
        norm_y_k_global = torch.norm(y_k_global_flat)

        # eps_div_guard is taken from the first param group; assume it's consistent.
        eps_div_guard = self.param_groups[0]['eps_div_guard']

        if norm_s_k_global > 0.0: # Avoid division by zero if s_k is zero (e.g., first step)
            # mu_DB = 1/L_k where L_k = ||y_k_global|| / ||s_k_global||
            # So, mu_DB = ||s_k_global|| / (||y_k_global|| + eps_div_guard)
            mu_DB = norm_s_k_global / (norm_y_k_global + eps_div_guard)
        # If norm_s_k_global is 0 (e.g. first step), mu_DB remains 0.0.

        # --- Update parameters and store state for the next step ---
        param_idx = 0
        for group in self.param_groups:
            lr_group = group['lr']

            # Calculate the 'mu' factor for the HVP term as in original optimizer's formula structure
            # effective_mu_for_hvp_term = mu_orig = lr_group * mu_DB
            effective_mu_for_hvp_term = lr_group * mu_DB

            for p in group['params']:
                if p.grad is None: # Should correspond to params not in params_with_grad
                    continue

                # Get per-parameter tensors for this step
                grad_p = current_grads_list[param_idx]       # g_k for current parameter p
                prev_dw_p = prev_dws_list[param_idx]         # s_k for p
                hvp_approx_p = y_k_list[param_idx]           # y_k for p (H_k * s_k approximation)

                # Gradient contribution: -lr * g_k
                grad_contribution = grad_p.mul(-lr_group)

                # Momentum contribution: s_k - (lr * mu_DB) * (H_k * s_k)
                # which is: prev_dw_p - effective_mu_for_hvp_term * hvp_approx_p
                # Using out-of-place torch.add to avoid modifying prev_dw_p if it's used elsewhere by reference
                momentum_contribution = torch.add(prev_dw_p, hvp_approx_p, alpha=-effective_mu_for_hvp_term)

                # Total update for w_k: Δw_k = grad_contribution + momentum_contribution
                update = grad_contribution.add(momentum_contribution)

                # Apply update to parameter p: w_{k+1} = w_k + Δw_k
                p.add_(update)

                # --- Store state for the next iteration (k+1) ---
                state = self.state[p]
                # Current update Δw_k becomes s_{k+1} for the next step
                state['dw_prev'] = update.clone(memory_format=torch.preserve_format)
                # Current gradient g_k becomes g_old for the next step's HVP: y_{k+1} = g_{k+1} - g_k
                state['grad_prev_step'] = grad_p.clone(memory_format=torch.preserve_format)
                state['step'] += 1

                param_idx += 1

        return loss