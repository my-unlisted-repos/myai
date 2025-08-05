import warnings
import torch
import torch.optim as optim
from collections import deque
import math

class HistoryWhiteningSVD(optim.Optimizer):
    def __init__(self, params, lr=1e-3, n_history=10, damping=1e-5,
                 update_precond_freq=1, beta_momentum=0.0, eps_svd_stability=1e-7):
        """
        Initialize the Historical Whitening Optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): Learning rate. Default: 1e-3.
            n_history (int, optional): Number of past gradient steps to store for whitening. Default: 10.
            damping (float, optional): Damping term (delta) for regularizing the covariance estimate.
                                       P = ( (1/k)HH^T + delta*I )^{-1/2}. Default: 1e-5.
            update_precond_freq (int, optional): Frequency (in steps) to recompute the preconditioner (SVD).
                                                 Default: 1 (every step).
            beta_momentum (float, optional): Beta for momentum on preconditioned gradients (0.0 means no momentum).
                                             Default: 0.0.
            eps_svd_stability (float, optional): Small epsilon added to singular values if using direct S^-1
                                                 (not used with current damping formula but kept for alternatives).
                                                 Default: 1e-7.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= damping:
            raise ValueError(f"Invalid damping term: {damping}")
        if not n_history >= 1:
            raise ValueError(f"n_history must be at least 1: {n_history}")
        if not update_precond_freq >= 1:
            raise ValueError(f"update_precond_freq must be at least 1: {update_precond_freq}")
        if not 0.0 <= beta_momentum < 1.0:
            raise ValueError(f"Invalid beta_momentum: {beta_momentum}")

        defaults = dict(lr=lr, n_history=n_history, damping=damping,
                        update_precond_freq=update_precond_freq,
                        beta_momentum=beta_momentum, eps_svd_stability=eps_svd_stability)
        super().__init__(params, defaults)

        # Initialize state for each parameter group
        for group in self.param_groups:
            group['grad_history_flat'] = deque(maxlen=group['n_history'])
            group['U_matrix'] = None
            group['S_transformed_diag'] = None
            group['step_count_group'] = 0 # For update_precond_freq per group
            if group['beta_momentum'] > 0:
                group['momentum_buffer_flat'] = None


    def _get_flat_grads_for_group(self, group):
        """Concatenate all gradients in a group into a single flat vector."""
        flat_grads = []
        for p in group['params']:
            if p.grad is None:
                # If a parameter has no gradient, fill with zeros of its shape.
                # This might happen if parts of the network are frozen or not used.
                flat_grads.append(torch.zeros_like(p.data).view(-1))
            else:
                flat_grads.append(p.grad.data.view(-1))
        if not flat_grads:
            return None
        return torch.cat(flat_grads)

    def _set_updates_for_group(self, group, update_flat_vec):
        """Distribute a flat update vector back to the parameters in a group."""
        offset = 0
        for p in group['params']:
            numel = p.data.numel()
            if numel == 0: continue # Skip empty parameters

            # Get the segment of the flat update vector for the current parameter
            param_update_flat = update_flat_vec[offset : offset + numel]

            # Reshape and apply the update
            p.data.add_(param_update_flat.view_as(p.data), alpha=-group['lr'])
            offset += numel
        if offset != update_flat_vec.numel():
            raise ValueError("Size mismatch when distributing flat updates back to parameters.")


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): # Ensure gradients are computed if closure needs them
                loss = closure()

        for group in self.param_groups:
            group['step_count_group'] += 1

            current_flat_grad = self._get_flat_grads_for_group(group)
            if current_flat_grad is None: # No params with grads in this group
                continue

            # Store current flat gradient in history
            # Detach to ensure we store values, not computation graph nodes
            group['grad_history_flat'].append(current_flat_grad.clone().detach())

            k_history = len(group['grad_history_flat'])

            # Determine if preconditioner should be updated
            update_now = (group['step_count_group'] % group['update_precond_freq'] == 0)

            if update_now and k_history >= 1: # Min 1 grad needed, ideally n_history
                # Form H matrix: d_group x k_history
                # Gradients are already flat, stack them as columns
                H_matrix = torch.stack(list(group['grad_history_flat']), dim=1)

                # Effective n for SVD (can be less than n_history if buffer not full)
                # The SVD is on H (d x k_history). U will be d x k_history, S will be k_history x k_history
                try:
                    U, S_singular_values, _ = torch.linalg.svd(H_matrix, full_matrices=False)

                    # Store U
                    group['U_matrix'] = U

                    # Compute transformed singular values for W = U * diag(vals) * U^T
                    # P = ( (1/k_history)HH^T + delta*I )^{-1/2}
                    # Eigenvalues of (1/k_history)HH^T are (S_singular_values^2 / k_history)
                    # So transformed singular values are 1.0 / sqrt( (S_singular_values^2 / k_history) + damping )
                    s_sq_scaled = (S_singular_values**2 / k_history) + group['damping']

                    # Add small epsilon before sqrt for stability, though damping should handle most.
                    # Using max with a small value to prevent sqrt of tiny negatives if s_sq_scaled is ~0
                    s_sq_stabilized = torch.max(s_sq_scaled, torch.full_like(s_sq_scaled, group['eps_svd_stability']/100))

                    group['S_transformed_diag'] = 1.0 / torch.sqrt(s_sq_stabilized)

                except torch.linalg.LinAlgError as e:
                    # SVD failed (e.g., NaN/inf gradients, or H_matrix is zero)
                    # Fallback: disable preconditioner for this group for this step
                    warnings.warn(f"Warning: SVD failed for a parameter group: {e}. Disabling preconditioning for this step.")
                    group['U_matrix'] = None
                    group['S_transformed_diag'] = None


            preconditioned_flat_grad = current_flat_grad
            if group['U_matrix'] is not None and group['S_transformed_diag'] is not None:
                # Apply preconditioner P g = U diag(S_transformed_diag) U^T g
                # Ensure S_transformed_diag has the correct dimensions for broadcasting if needed
                # U is (d, k_svd), S_transformed_diag is (k_svd), U.T @ g is (k_svd)
                # k_svd = min(d_group, k_history)

                try:
                    ut_g = group['U_matrix'].T @ current_flat_grad
                    scaled_ut_g = group['S_transformed_diag'] * ut_g
                    preconditioned_flat_grad = group['U_matrix'] @ scaled_ut_g
                except RuntimeError as e:
                    # This can happen if shapes mismatch, e.g. if SVD output k_svd is 0
                    print(f"Warning: Failed to apply preconditioner: {e}. Using raw gradient.")
                    group['U_matrix'] = None # Disable for next time until recomputed
                    group['S_transformed_diag'] = None
                    preconditioned_flat_grad = current_flat_grad


            # Optional Momentum
            if group['beta_momentum'] > 0.0:
                if group.get('momentum_buffer_flat') is None:
                    group['momentum_buffer_flat'] = torch.zeros_like(preconditioned_flat_grad)

                buf = group['momentum_buffer_flat']
                buf.mul_(group['beta_momentum']).add_(preconditioned_flat_grad, alpha=1.0) # In-place version if possible
                # buf = group['beta_momentum'] * buf + preconditioned_flat_grad # Simpler, might create new tensor
                final_update_vec = buf
            else:
                final_update_vec = preconditioned_flat_grad

            # Apply updates to parameters in the group
            self._set_updates_for_group(group, final_update_vec)

        return loss