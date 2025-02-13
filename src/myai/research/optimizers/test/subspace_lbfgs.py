import torch
from torch.optim import Optimizer
import numpy as np

class RandomSubspaceLBFGS(Optimizer):
    """
    L-BFGS in a random subspace (for now).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (not directly used by L-BFGS, but kept for interface compatibility)
        subspace_dim (int): dimension of the random subspace
        history_size (int): history size for L-BFGS
        betas (Tuple[float, float]): coefficients used for moving average in subspace basis update (beta1, beta2)
        full_space_update (bool): If True, perform a standard gradient descent step in the full space
                                  after the subspace L-BFGS step. This can help escape poor subspaces.
        full_space_lr (float): Learning rate for the full space gradient descent step (if full_space_update=True).

    TODO:
        change subspace to mix of EMAs
    """

    def __init__(self, params, lr=1.0, subspace_dim=20, history_size=10, betas=(0.9, 0.999), full_space_update=False, full_space_lr=0.01):
        defaults = dict(lr=lr, subspace_dim=subspace_dim, history_size=history_size, betas=betas,
                        full_space_update=full_space_update, full_space_lr=full_space_lr)
        super().__init__(params, defaults)

        for group in self.param_groups:
            group['subspace_basis'] = None
            group['m_flat'] = None
            group['s_history'] = []
            group['y_history'] = []
            group['rho_history'] = []
            group['step_count'] = 0
            group['prev_grad_flat'] = None


    def _create_random_subspace(self, flat_params, subspace_dim, device, m_flat_avg=None):
        n_params = len(flat_params)
        if subspace_dim >= n_params:
            return torch.eye(n_params, device=device) # Full space if subspace_dim is too large

        if m_flat_avg is None: # Initial random subspace
            basis = torch.randn(n_params, subspace_dim, device=device)
        else: # Moving average subspace basis update
            random_basis_part = torch.randn(n_params, subspace_dim, device=device)
            basis = self.defaults['betas'][0] * m_flat_avg + (1 - self.defaults['betas'][0]) * random_basis_part

        basis_normalized, _ = torch.linalg.qr(basis) # Orthonormalize using QR decomposition
        return basis_normalized

    def _project_grad_to_subspace(self, flat_grad, subspace_basis):
        return torch.matmul(subspace_basis.transpose(0, 1), flat_grad)

    def _lift_update_to_full_space(self, subspace_update, subspace_basis):
        return torch.matmul(subspace_basis, subspace_update)

    def _lbfgs_two_loop_recursion(self, grad_subspace, s_history, y_history, rho_history, H_diag=None):
        history_size = len(s_history)
        alpha = []
        q = grad_subspace.clone()

        for i in reversed(range(history_size)):
            rho_i = rho_history[i]
            s_i = s_history[i]
            y_i = y_history[i]
            alpha_i = rho_i * torch.dot(s_i, q)
            alpha.append(alpha_i)
            q.add_(-alpha_i, y_i)

        if H_diag is None:
            gamma = 1.0
            if y_history and s_history:
                s_k = s_history[-1]
                y_k = y_history[-1]
                gamma = torch.dot(s_k, y_k) / torch.dot(y_k, y_k) if torch.dot(y_k, y_k) > 0 else 1.0 # Avoid division by zero
            r = q * gamma
        else:
            r = H_diag * q

        for i in range(history_size):
            rho_i = rho_history[i]
            s_i = s_history[i]
            y_i = y_history[i]
            beta_i = rho_i * torch.dot(y_i, r)
            r.add_(s_i, alpha = alpha[history_size - 1 - i] - beta_i)

        return r

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            flat_params = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    flat_params.append(p.data.reshape(-1))
            if not params_with_grad:
                continue

            flat_params = torch.cat(flat_params)
            flat_grad = torch.cat([g.reshape(-1) for g in grads])
            state = group


            # 1. Subspace basis update (Moving Average - optional, can be removed for simpler version)
            if state['m_flat'] is None:
                state['m_flat'] = torch.zeros_like(flat_params)

            state['m_flat'] = self.defaults['betas'][1] * state['m_flat'] + (1 - self.defaults['betas'][1]) * flat_grad


            # 2. Create or update subspace basis
            if state['subspace_basis'] is None:
                state['subspace_basis'] = self._create_random_subspace(flat_params, group['subspace_dim'], flat_params.device, None)
            else:
                state['subspace_basis'] = self._create_random_subspace(flat_params, group['subspace_dim'], flat_params.device, state['m_flat'])


            # 3. Project gradient to subspace
            grad_subspace = self._project_grad_to_subspace(flat_grad, state['subspace_basis'])


            # 4. L-BFGS in subspace
            s_history = state['s_history']
            y_history = state['y_history']
            rho_history = state['rho_history']
            history_size = group['history_size']


            if state['step_count'] > 0:
                s = self.subspace_update_direction # From previous step
                y = grad_subspace - state['prev_grad_subspace']

                s_dot_y = torch.dot(s, y)
                if s_dot_y > 1e-10: # Positive curvature condition
                    rho = 1.0 / s_dot_y
                    state['s_history'].append(s)
                    state['y_history'].append(y)
                    state['rho_history'].append(rho)
                    if len(state['s_history']) > history_size:
                        state['s_history'].pop(0)
                        state['y_history'].pop(0)
                        state['rho_history'].pop(0)
                else:
                    pass # Skip update if curvature not positive


            self.subspace_update_direction = self._lbfgs_two_loop_recursion(grad_subspace, s_history, y_history, rho_history)
            self.subspace_update_direction.mul_(-group['lr']) # Apply learning rate (for interface, L-BFGS often doesn't use LR directly)


            # 5. Lift update to full space
            full_space_update_direction = self._lift_update_to_full_space(self.subspace_update_direction, state['subspace_basis'])


            # 6. Apply update
            param_idx = 0
            for p in params_with_grad:
                numel = p.numel()
                update_chunk = full_space_update_direction[param_idx:param_idx + numel].reshape(p.shape)
                p.data.add_(update_chunk)
                param_idx += numel


            # 7. Full space gradient step (optional) - to escape bad subspaces
            if group['full_space_update']:
                param_idx = 0
                full_space_grad_update = -group['full_space_lr'] * flat_grad
                for p in params_with_grad:
                    numel = p.numel()
                    update_chunk = full_space_grad_update[param_idx:param_idx + numel].reshape(p.shape)
                    p.data.add_(update_chunk)
                    param_idx += numel


            # Store current subspace gradient for next step's L-BFGS update
            state['prev_grad_subspace'] = grad_subspace.clone()
            state['step_count'] += 1


        return loss