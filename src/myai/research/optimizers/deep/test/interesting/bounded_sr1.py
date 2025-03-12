import torch
from torch.optim import Optimizer

class MBSROptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, history_size=10, norm_growth_bound=1.5, pos = 1.1, neg = 0.9):
        defaults = dict(lr=lr, history_size=history_size, norm_growth_bound=norm_growth_bound)
        super().__init__(params, defaults)

        for group in self.param_groups:
            group['Hk'] = None # Hessian approximation, will be initialized per parameter group
            group['s_history'] = []
            group['y_history'] = []

        self.pos = pos
        self.neg = neg
        self.adapt_value = 1

    def _initial_hessian_approx(self, params):
        # Initialize with Identity matrix (or scaled Identity if needed)
        device = params[0].device
        flat_params = torch.cat([p.flatten() for p in params])
        return torch.eye(flat_params.numel(), dtype=flat_params.dtype, device=device)

    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr'] * self.adapt_value
            norm_growth_bound = group['norm_growth_bound']
            history_size = group['history_size']

            params_with_grad = []
            d_p_list = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            if not params_with_grad:
                continue

            grads = [d_p.flatten() for d_p in d_p_list]
            current_grad = torch.cat(grads)
            params = [p.flatten() for p in params_with_grad]
            current_params = torch.cat(params)

            if group['Hk'] is None:
                group['Hk'] = self._initial_hessian_approx(params_with_grad)

            Hk = group['Hk']

            # 1. Compute update direction
            update_vector = -torch.matmul(Hk, current_grad)

            # 2. Bounded Norm Growth
            current_param_norm = torch.linalg.norm(current_params) # Parameter norm
            update_norm = torch.linalg.norm(update_vector)
            allowed_update_norm = norm_growth_bound * current_param_norm

            if update_norm > allowed_update_norm:
                scale_factor = allowed_update_norm / update_norm
                update_vector = update_vector * scale_factor

            # 3. Tentative parameter update
            with torch.no_grad():
                param_idx = 0
                for p in params_with_grad:
                    numel = p.numel()
                    p.add_(update_vector[param_idx:param_idx + numel].reshape(p.shape), alpha=lr) # Apply LR here to update *p*
                    param_idx += numel

            # 4. Second Closure Evaluation to get g_{k+1}'
            with torch.enable_grad(): loss2 = closure()
            if loss2 < loss: self.adapt_value *= self.pos
            else: self.adapt_value *= self.neg

            next_grads = []
            for p in params_with_grad:
                if p.grad is not None: # Gradients should be recomputed by closure
                    next_grads.append(p.grad.flatten())
                else:
                    raise RuntimeError("Gradients were not recomputed by the second closure.")
            next_grad = torch.cat(next_grads)

            # 5. Calculate differences s_k and y_k
            s_k = update_vector * lr # s_k = p_{k+1}' - p_k = Δp_k, and we applied LR to update in step 3.
            y_k = next_grad - current_grad

            # 6. SR1 Update of Hessian Approximation (using latest (s_k, y_k))
            sTy = torch.dot(s_k, y_k)
            if sTy > 1e-8: # Check for positive curvature (avoid division by zero/instability)
                v_k = y_k - torch.matmul(Hk, s_k)
                rho_k = 1.0 / sTy
                Hk_update = rho_k * torch.outer(v_k, v_k)
                Hk.add_(Hk_update)
                group['Hk'] = Hk # Update Hessian in param_group

            # 7. History Update (simple history storage - not actively used for Hessian update in this version, but can be used for analysis/future extensions)
            group['s_history'].append(s_k.detach()) # Store detached versions
            group['y_history'].append(y_k.detach())
            if len(group['s_history']) > history_size:
                group['s_history'].pop(0)
                group['y_history'].pop(0)

        return loss