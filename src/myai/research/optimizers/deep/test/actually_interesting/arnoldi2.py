# pylint:disable = not-callable
import torch
from torch.optim import Optimizer

class ArnoldiGD(Optimizer):
    """diabolical"""
    def __init__(self, params, lr=1e-3, krylov_dim=5, weight_decay=0, lambda_reg=1e-6, fd_epsilon=1e-4):
        defaults = dict(lr=lr, weight_decay=weight_decay, krylov_dim=krylov_dim, lambda_reg=lambda_reg, fd_epsilon=fd_epsilon)
        super().__init__(params, defaults)

    def _hessian_vector_product_fd(self, vector, params, closure):
        """Computes Hessian-vector product using finite differences."""
        params_vec = torch.nn.utils.parameters_to_vector(params)
        v_vec = vector
        fd_epsilon = self.defaults['fd_epsilon']

        # Perturb parameters in the direction of v
        params_plus_vec = params_vec + fd_epsilon * v_vec
        params_minus_vec = params_vec - fd_epsilon * v_vec

        def get_grads_at(params_vec_perturbed):
            torch.nn.utils.vector_to_parameters(params_vec_perturbed, params)
            with torch.enable_grad(): _ = closure() # Re-evaluate loss with perturbed parameters
            perturbed_grads = [p.grad for p in params]
            return torch.nn.utils.parameters_to_vector(perturbed_grads)

        grad_plus = get_grads_at(params_plus_vec)
        grad_minus = get_grads_at(params_minus_vec)

        hv_approx = (grad_plus - grad_minus) / (2 * fd_epsilon)
        return hv_approx


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if len(self.param_groups) > 1: raise NotImplementedError(f'only 1 param group supported, got {len(self.param_groups)}')
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state.append(self.state[p])

            if not params_with_grad:
                continue

            lr = group['lr']
            weight_decay = group['weight_decay']
            krylov_dim = group['krylov_dim']
            lambda_reg = group['lambda_reg']

            # Vectorize gradients and parameters
            grad_vec = torch.nn.utils.parameters_to_vector(grads)
            param_vec = torch.nn.utils.parameters_to_vector(params_with_grad).data # .data to avoid inplace op error


            # 1. Arnoldi Iteration
            q = []
            H = torch.zeros((krylov_dim, krylov_dim), dtype=grad_vec.dtype, device=grad_vec.device)

            q1 = grad_vec / torch.linalg.norm(grad_vec)
            q.append(q1)

            current_params = params_with_grad

            for k in range(krylov_dim - 1):
                qk = q[-1]
                v = self._hessian_vector_product_fd(qk, current_params, closure)

                for j in range(k + 1):
                    H[j, k] = torch.dot(q[j], v)
                    v = v - H[j, k] * q[j]

                h_norm = torch.linalg.norm(v)
                if h_norm == 0: # Krylov subspace spanned early
                    H = H[:k+1, :k+1] # Truncate Hessenberg matrix
                    break
                H[k+1, k] = h_norm
                q_next = v / h_norm
                q.append(q_next)

            Q = torch.stack(q, dim=1) # Q = [q1, q2, ..., q_m], shape (N_params, m)
            Qm = Q # In case Arnoldi stopped early, Qm = Q will handle it correctly

            # 2. Solve Projected System
            e1 = torch.zeros(Qm.shape[1], dtype=grad_vec.dtype, device=grad_vec.device)
            e1[0] = 1.0
            Hm = H[:Qm.shape[1], :Qm.shape[1]] # Truncate Hessenberg matrix to m x m or smaller if early stop

            A = Hm + lambda_reg * torch.eye(Hm.shape[0], dtype=grad_vec.dtype, device=grad_vec.device)
            b = -torch.linalg.norm(grad_vec) * e1

            try:
                y = torch.linalg.solve(A, b) # Solve (Hm + lambda_reg*I)y = -||g||e1
                update_direction = Qm @ y #  d = Q_m @ y
            except RuntimeError as e: # Handle potential linear solver issues (e.g., singular matrix)
                print(f"Linear Solver Warning: {e}. Falling back to projected gradient.")
                update_direction = Qm @ (Qm.T @ grad_vec) # Fallback: project gradient

            # 3. Parameter Update
            if weight_decay != 0:
                update_direction.add_(param_vec, alpha=weight_decay) # Weight decay on parameters *before* update

            param_vec.add_(update_direction, alpha=lr) # Apply update, negation for gradient descent

            # 4. Distribute updated parameters back to model
            torch.nn.utils.vector_to_parameters(param_vec, params_with_grad)

        return loss