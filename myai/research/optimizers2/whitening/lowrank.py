import math
import warnings
from collections import deque

import torch
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# Helper functions for flattening/unflattening gradients/parameters
def _flatten_gradients(params):
    """Flatten gradients from a list of parameters into a single tensor."""
    grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(0) # Or handle appropriately
    return torch.cat(grads)

def _flatten_params(params):
    """Flatten parameters from a list of parameters into a single tensor."""
    return torch.cat([p.detach().reshape(-1) for p in params])

def _unflatten_like(flat_tensor, params_like):
    """Unflatten a tensor according to the shapes of parameters in params_like."""
    grads_out = []
    current_pos = 0
    for p in params_like:
        numel = p.numel()
        if flat_tensor.numel() > 0: # Ensure flat_tensor is not empty
           chunk = flat_tensor[current_pos : current_pos + numel]
           grads_out.append(chunk.reshape_as(p))
        else: # Handle empty gradient case (e.g. no params had grad)
           grads_out.append(torch.zeros_like(p))
        current_pos += numel
    if current_pos != flat_tensor.numel():
         raise ValueError(f"Size mismatch during unflattening: expected {current_pos}, got {flat_tensor.numel()}")
    return grads_out

class LoRAP(optim.Optimizer):
    """
    Implements the Low Rank Approximation Preconditioner (LoRAP) algorithm.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        history_size (int, optional): Maximum size of the gradient history (default: 20).
        rank (int, optional): Rank of the subspace for approximation. If None, uses full history size. (default: None)
        eps (float, optional): Term added to the diagonal of the covariance matrix for numerical stability (default: 1e-6).
        update_freq (int, optional): Frequency (in steps) for updating the subspace basis Q (default: 10).
        beta (float, optional): Optional momentum parameter (similar to SGD momentum). If 0, no momentum. (default: 0)
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr=1e-3, history_size=20, rank=None, eps=1e-6, update_freq=10, beta=0.0, weight_decay=0.0):

        defaults = dict(lr=lr, history_size=history_size, rank=rank or history_size,
                        eps=eps, update_freq=update_freq, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # State initialization (per optimizer instance, not per param group)
        self._params_list = []
        for group in self.param_groups:
            self._params_list.extend(group['params'])

        self._history = deque(maxlen=history_size)
        self._Q = None # Orthonormal basis matrix (N x r)
        self._step = 0
        self._rank = rank or history_size

        # Momentum state (optional)
        if beta > 0:
            self.state['momentum_buffer'] = torch.zeros_like(_flatten_params(self._params_list))


    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1

        # --- 1. Get current flattened gradient and apply weight decay ---
        flat_grad = _flatten_gradients(self._params_list)

        if len(flat_grad) == 0: # Handle cases with no gradients
             warnings.warn("Warning: No gradients found for any parameters. Skipping step.")
             return loss

        # Apply weight decay if specified
        if self.defaults['weight_decay'] != 0:
            flat_params = _flatten_params(self._params_list)
            flat_grad = flat_grad.add(flat_params, alpha=self.defaults['weight_decay'])

        device = flat_grad.device

        # --- 2. Update History ---
        self._history.append(flat_grad.clone())

        # --- Check if history is sufficiently populated ---
        if len(self._history) < self._rank or len(self._history) < 2: # Need at least rank vectors and at least 2 for covariance
            # Fallback to SGD or SGD with momentum if history is too short
            update_direction = flat_grad
            if self.defaults['beta'] > 0:
                 buf = self.state['momentum_buffer']
                 buf.mul_(self.defaults['beta']).add_(update_direction)
                 update_direction = buf
            # Perform update
            flat_params = _flatten_params(self._params_list)
            flat_params.add_(update_direction, alpha=-self.defaults['lr'])
            # Unflatten updated parameters back
            new_param_chunks = _unflatten_like(flat_params, self._params_list)
            for p, p_new in zip(self._params_list, new_param_chunks):
                 p.copy_(p_new) # Use copy_ to update in place
            return loss

        # --- 3. Update Subspace Basis Q (periodically) ---
        if self._Q is None or self._step % self.defaults['update_freq'] == 0:
            # Stack gradients history into a matrix (m x N)
            G_hist_t = torch.stack(list(self._history), dim=0) # Shape: (m, N)

            # Use SVD to find orthonormal basis for the column space (which corresponds to gradient directions)
            # We compute SVD on G^T (N x m) to get U (N x k), S (k), Vh (k x m) where k = min(N, m)
            # The columns of U are the basis vectors Q.
            try:
                # Note: torch.linalg.svd computes U, S, Vh such that A = U S Vh
                U, S, Vh = torch.linalg.svd(G_hist_t.t(), full_matrices=False) # U is N x min(N,m)

                # Select top 'rank' singular vectors
                rank_to_use = min(self._rank, S.size(0), len(self._history)) # Ensure rank is valid

                # Filter near-zero singular values to avoid numerical issues
                non_zero_s_mask = S > 1e-8 # Tolerance for singular values
                rank_to_use = min(rank_to_use, non_zero_s_mask.sum().item())

                if rank_to_use == 0:
                    warnings.warn("Warning: Rank of history matrix is 0 at step. Falling back to SGD.")
                    self._Q = None # Reset Q if rank becomes 0
                else:
                    self._Q = U[:, :rank_to_use] # Shape: (N, r)
                    # print(f"Step {self._step}: Updated Q with rank {self._Q.shape[1]}") # Debug

            except torch.linalg.LinAlgError as e:
                warnings.warn(f"Warning: SVD failed at step: {e}. Falling back to SGD.")
                self._Q = None # Reset Q on SVD failure

        # --- If Q is still None (e.g., rank 0 or SVD failed), fallback ---
        if self._Q is None:
            update_direction = flat_grad
            if self.defaults['beta'] > 0:
                buf = self.state['momentum_buffer']
                buf.mul_(self.defaults['beta']).add_(update_direction)
                update_direction = buf
            flat_params = _flatten_params(self._params_list)
            flat_params.add_(update_direction, alpha=-self.defaults['lr'])
            new_param_chunks = _unflatten_like(flat_params, self._params_list)
            for p, p_new in zip(self._params_list, new_param_chunks):
                 p.copy_(p_new)
            return loss


        # --- 4. Project Gradients onto Subspace ---
        # Current gradient projection: g'_k = Q^T g_k
        g_proj = self._Q.t() @ flat_grad # Shape: (r, 1)

        # Historical gradients projection: G'_hist = Q^T G_hist
        # G_hist is (N x m), so G'_hist is (r x m)
        G_hist_t = torch.stack(list(self._history), dim=0) # Shape: (m, N)
        G_proj_t = G_hist_t @ self._Q # Shape: (m, r) - Project each row (gradient)

        # --- 5. Compute Covariance within Subspace ---
        # G_proj_t is (m, r). We want covariance of the r projected dimensions.
        current_m = G_proj_t.shape[0]
        if current_m < 2: # Need at least 2 samples for covariance
             # Fallback if not enough samples yet for covariance
             p_proj = g_proj # Use identity preconditioner in subspace
        else:
             means = G_proj_t.mean(dim=0) # Shape: (r,)
             centered_G_proj_t = G_proj_t - means # Shape: (m, r)
             # Covariance C = (1/(m-1)) * centered_G_proj^T @ centered_G_proj
             # Note the transpose compared to standard Cov formula because data points are rows here
             C = (centered_G_proj_t.t() @ centered_G_proj_t) / (current_m - 1) # Shape: (r, r)


             # --- 6. Solve Preconditioning System in Subspace ---
             # Solve (C + eps*I) p'_k = g'_k
             try:
                 # Add damping
                 C_damped = C + torch.eye(C.shape[0], device=device) * self.defaults['eps']
                 # Solve using Cholesky decomposition for stability and efficiency with SPD matrices
                 L = torch.linalg.cholesky(C_damped)
                 p_proj = torch.cholesky_solve(g_proj.unsqueeze(-1), L).squeeze(-1) # Shape: (r,)
             except torch.linalg.LinAlgError:
                 # Fallback if Cholesky fails (e.g., C is ill-conditioned despite damping)
                 warnings.warn("Warning: Cholesky solve failed at step. Using damped identity preconditioner in subspace.")
                 p_proj = g_proj / (torch.diag(C).mean() + self.defaults['eps']) # Simple diagonal scaling fallback


        # --- 7. Map Back to Full Space ---
        # p_k = Q p'_k
        preconditioned_grad = self._Q @ p_proj # Shape: (N,)

        # --- 8. Apply Momentum (Optional) ---
        update_direction = preconditioned_grad
        if self.defaults['beta'] > 0:
            buf = self.state['momentum_buffer']
            buf.mul_(self.defaults['beta']).add_(update_direction) # Update momentum buffer
            update_direction = buf # Use momentum buffer as the final update direction

        # --- 9. Apply Update ---
        # theta_{k+1} = theta_k - lr * p_k
        flat_params = _flatten_params(self._params_list)
        flat_params.add_(update_direction, alpha=-self.defaults['lr'])

        # Unflatten updated parameters back into model parameters
        new_param_chunks = _unflatten_like(flat_params, self._params_list)
        for p, p_new in zip(self._params_list, new_param_chunks):
            p.copy_(p_new) # Use copy_ for in-place update compatible with optimizer state

        return loss


# Helper function to get parameters and gradients for a group
# Flattens them into single vectors
def _get_flat_params_grads(param_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Flattens parameters and their gradients for a list of tensors."""
    original_shapes = [p.shape for p in param_list]
    params_flat = parameters_to_vector(param_list)

    # Ensure gradients exist and are usable
    valid_grads = []
    for p in param_list:
        if p.grad is not None:
            valid_grads.append(p.grad.detach().clone()) # Clone to avoid modifying original grads
        else:
            # If a parameter has no gradient, treat it as zero
            valid_grads.append(torch.zeros_like(p))

    if not valid_grads:
         # Handle case where no parameters in the group have gradients
         grads_flat = torch.zeros_like(params_flat)
    else:
        grads_flat = parameters_to_vector(valid_grads)

    return params_flat, grads_flat, original_shapes

# Helper function to set updated flat parameters back to original tensors
def _set_flat_params(param_list: list[torch.Tensor], params_flat: torch.Tensor):
    """Sets parameter values from a flat vector."""
    vector_to_parameters(params_flat, param_list)


class CompressedAdam(optim.Optimizer):
    r"""Implements Adam algorithm with compressed low-rank second moment estimation.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        rank (int): Rank of the low-rank approximation for the second moment matrix.
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) - Note: AMSGrad is not implemented for the low-rank part yet.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, rank=10, amsgrad=False):
        if amsgrad:
            print("Warning: AMSGrad is not fully implemented for the low-rank component in CompressedAdam.")
            # Need to track max(Lambda) similarly to max(v_hat) if implemented.

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, rank=rank, amsgrad=amsgrad)
        super(CompressedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CompressedAdam, self).__setstate__(state)
        # Ensure backward compatibility for amsgrad state
        for group in self.param_groups:
            group.setdefault('amsgrad', False)


    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sq_qs = [] # Stores Q matrices
            exp_avg_sq_lambdas = [] # Stores Lambda vectors (diagonal of Lambda matrix)
            state_steps = []
            beta1, beta2 = group['betas']
            rank = group['rank']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad'] # Currently unused for low-rank part

            # --- Parameter Group Setup ---
            # Gather parameters and gradients for this group
            current_params = [p for p in group['params'] if p.grad is not None]
            if not current_params:
                continue # Skip group if no gradients

            # Flatten parameters and gradients for this group
            flat_params, flat_grads, original_shapes = _get_flat_params_grads(current_params)
            param_dim = flat_params.numel()

            # Check if rank is valid
            if rank >= param_dim:
                # Fallback to standard Adam logic might be needed here,
                # or just proceed knowing Q will be full rank PxP (inefficient).
                # For simplicity, we'll proceed but it might be slow/numerically unstable.
                # A better approach would be to cap rank = param_dim - 1 or fallback.
                current_rank = min(rank, param_dim - 1) if param_dim > 0 else 0
            else:
                current_rank = rank

            # --- State Initialization ---
            state = self.state[flat_params] # Use flat_params as the key
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values (m)
                state['exp_avg'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)
                # Low-rank factors for the raw second moment (V)
                # Q: P x k matrix (eigenvectors)
                state['Q'] = torch.zeros((param_dim, current_rank), dtype=flat_params.dtype, device=flat_params.device)
                # Lambda: k vector (eigenvalues)
                state['Lambda'] = torch.zeros(current_rank, dtype=flat_params.dtype, device=flat_params.device)
                # AMSGrad max state (if applicable, needs adaptation)
                # if amsgrad: state['max_exp_avg_sq'] = torch.zeros_like(flat_params, memory_format=torch.preserve_format)


            # --- Adam Update Steps ---
            state['step'] += 1
            step = state['step']
            m_t = state['exp_avg']
            Q_prev = state['Q']
            Lambda_prev = state['Lambda'] # This is a vector

            # Apply weight decay
            if weight_decay != 0:
                flat_grads = flat_grads.add(flat_params, alpha=weight_decay)

            # Update biased first moment estimate (m_t)
            m_t.mul_(beta1).add_(flat_grads, alpha=1 - beta1)

            # --- Update Low-Rank Second Moment (V_t) ---
            # Construct the matrix A = [sqrt(β₂) * Q_{t-1} * sqrt(Λ_{t-1}), sqrt(1 - β₂) * g_t]
            # Note: Λ is diagonal, so sqrt(Λ) is element-wise sqrt on the vector Lambda_prev
            sqrt_lambda_prev = Lambda_prev.sqrt()
            term1 = math.sqrt(beta2) * Q_prev * sqrt_lambda_prev.unsqueeze(0) # P x k
            term2 = math.sqrt(1 - beta2) * flat_grads.unsqueeze(1) # P x 1

            A = torch.cat([term1, term2], dim=1) # P x (k+1)

            # Perform SVD on A
            try:
                # Using torch.linalg.svd which is generally preferred
                # full_matrices=False is important for efficiency with tall matrices
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            except torch.linalg.LinAlgError as e:
                 print(f"Warning: SVD failed on step {step}. Skipping second moment update. Error: {e}")
                 # Keep previous Q, Lambda if SVD fails
                 Q_t = Q_prev
                 Lambda_t = Lambda_prev # vector
            else:
                 # Update Q_t and Lambda_t (top k components)
                 Q_t = U[:, :current_rank] # P x k
                 # S contains singular values, Lambda contains eigenvalues (S^2)
                 Lambda_t = S[:current_rank].pow(2) # k vector

            # Update state
            state['Q'] = Q_t
            state['Lambda'] = Lambda_t

            # --- Bias Correction ---
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step # Used for V_hat

            # Calculate m_hat
            m_hat_t = m_t / bias_correction1

            # --- Compute Update Step using Low-Rank Preconditioner ---
            # We need Δp = V_hat_t^{-1/2} * m_hat_t
            # V_hat_t^{-1/2} ≈ sqrt(bias_correction2) * Q_t * diag(1 / (sqrt(λ_i) + ε)) * Q_t^T

            # Calculate the scaled inverse sqrt eigenvalues
            # Add eps inside sqrt for stability before division potentially by zero? No, Adam adds after.
            # Add eps to sqrt(lambda) before division
            lambda_sqrt = Lambda_t.sqrt()
            inv_lambda_stabilized = 1.0 / (lambda_sqrt + eps) # k vector

            # Project m_hat onto the subspace spanned by Q_t
            y = torch.matmul(Q_t.T, m_hat_t) # k vector

            # Scale within the subspace
            z = inv_lambda_stabilized * y # k vector

            # Project back to the full parameter space and apply bias correction scaling
            # delta_p = sqrt(bias_correction2) * torch.matmul(Q_t, z) # P vector
            # Correction based on Adam formula: update = lr * m_hat / (sqrt(v_hat) + eps)
            # Our delta_p should approximate m_hat / (sqrt(V_hat) + eps)
            # V_hat = V / bc2 => sqrt(V_hat) = sqrt(V)/sqrt(bc2) => 1/sqrt(V_hat) = sqrt(bc2)/sqrt(V)
            # So, V_hat^{-1/2} * m_hat approx = sqrt(bc2) * Q * diag(1/sqrt(L)) * Q^T * m_hat
            # The computation looks right, let's compute the denominator sqrt(V_hat) + eps effectively

            # Revisit the update rule: p_new = p_old - lr * m_hat / (sqrt(v_hat) + eps)
            # We want: p_new = p_old - lr * Precond * m_hat
            # Where Precond approx V_hat^{-1/2}
            # Precond = sqrt(bc2) * Q * diag(1/(sqrt(L) + eps_eff)) * Q.T
            # Let's directly compute the effective preconditioner applied to m_hat
            # Denominator term: D = sqrt(V_hat) + eps_matrix ? Complicated.
            # Let's use the update delta = Q * diag(1/(sqrt(Lambda_hat)+eps)) * Q.T * m_hat
            # Lambda_hat = Lambda_t / bias_correction2
            # sqrt(Lambda_hat) = sqrt(Lambda_t) / sqrt(bias_correction2)
            # Denom_eigen = sqrt(Lambda_t) / math.sqrt(bias_correction2) + eps

            denom_eigen_sqrt = Lambda_t.sqrt() / math.sqrt(bias_correction2)
            inv_sqrt_lambda_hat_stabilized = 1.0 / (denom_eigen_sqrt + eps)

            y_hat = torch.matmul(Q_t.T, m_hat_t) # k vector
            z_hat = inv_sqrt_lambda_hat_stabilized * y_hat # k vector
            update_direction = torch.matmul(Q_t, z_hat) # P vector

            # --- Apply Update ---
            # Update the flattened parameters
            flat_params.add_(update_direction, alpha=-lr)

            # --- Unflatten and Update Original Parameters ---
            _set_flat_params(current_params, flat_params)

        return loss

