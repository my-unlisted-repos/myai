import math
import warnings
from collections import defaultdict

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

# Utility functions for flattening/unflattening parameters might be needed
# depending on how state is managed across parameter groups.
# For simplicity here, we assume one parameter group or apply independently.
# PyTorch >= 1.10 has torch.nn.utils.convert_parameters.{vector_to_parameters, parameters_to_vector}

class HouseholderAdam(Optimizer):
    """
    Implements Adam algorithm with second moment estimation using
    a diagonal + low-rank approximation via Householder reflections.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        k (int, optional): rank of the low-rank approximation (number of
            Householder vectors) (default: 10)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
            (Not implemented here, would require tracking max exp_avg_sq)
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k=10, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0 <= k:
            raise ValueError("Invalid rank k: {}".format(k))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if amsgrad:
             warnings.warn("AMSGrad is not implemented for HouseholderAdam in this version.")
             # To implement AMSGrad, you'd need to track max_exp_avg_sq like in torch.optim.Adam


        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, k=k, amsgrad=amsgrad)
        super(HouseholderAdam, self).__init__(params, defaults)

        # Initialize group-specific states
        # We store flattened moments and householder vectors per group
        for group in self.param_groups:
            group_params = group['params']
            if not group_params:
                continue

            # Determine total size and device for flattened tensors
            first_param = group_params[0]
            device = first_param.device
            total_param_size = sum(p.numel() for p in group_params if p.requires_grad)

            group_state = self.state[f'group_{id(group)}'] # Use group id as key
            group_state['step'] = 0
            group_state['total_param_size'] = total_param_size
            group_state['device'] = device

            # Initialize flattened moments on the correct device
            group_state['exp_avg'] = torch.zeros(total_param_size, device=device)
            group_state['exp_avg_sq'] = torch.zeros(total_param_size, device=device)

            # Initialize Householder state
            group_state['k'] = group['k']
            if group_state['k'] > 0:
                group_state['householder_vectors'] = [None] * group_state['k']
                group_state['vector_norms_sq'] = torch.zeros(group_state['k'], device=device)
                group_state['vector_index'] = 0 # Pointer to the next slot in the circular buffer

    def __setstate__(self, state):
        super(HouseholderAdam, self).__setstate__(state)
        # Ensure backward compatibility for state dicts if needed
        # For example, initializing group-specific state if it's missing
        for group in self.param_groups:
            group_id = f'group_{id(group)}'
            if group_id not in self.state:
                 # Re-initialize group state if loading an older state dict
                 # (This part might need adjustment based on saving/loading logic)
                 print(f"Re-initializing state for group {id(group)}")
                 group_params = group['params']
                 if not group_params: continue
                 first_param = group_params[0]
                 device = first_param.device
                 total_param_size = sum(p.numel() for p in group_params if p.requires_grad)

                 group_state = self.state[group_id]
                 group_state['step'] = 0
                 group_state['total_param_size'] = total_param_size
                 group_state['device'] = device
                 group_state['exp_avg'] = torch.zeros(total_param_size, device=device)
                 group_state['exp_avg_sq'] = torch.zeros(total_param_size, device=device)
                 group_state['k'] = group['k']
                 if group_state['k'] > 0:
                     group_state['householder_vectors'] = [None] * group_state['k']
                     group_state['vector_norms_sq'] = torch.zeros(group_state['k'], device=device)
                     group_state['vector_index'] = 0

    @torch.no_grad()
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
            group_params = [p for p in group['params'] if p.grad is not None]
            if not group_params:
                continue

            # Retrieve group-specific state
            group_id = f'group_{id(group)}'
            group_state = self.state[group_id]

            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            k = group_state['k']

            # Increment step
            group_state['step'] += 1
            bias_correction1 = 1.0 - beta1 ** group_state['step']
            bias_correction2 = 1.0 - beta2 ** group_state['step']

            # --- 1. Flatten Gradients and Apply Weight Decay ---
            flat_grads = []
            for p in group_params:
                if p.grad.is_sparse:
                     raise RuntimeError('HouseholderAdam does not support sparse gradients')
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                flat_grads.append(grad.reshape(-1))

            if not flat_grads: # Should not happen if group_params is not empty
                continue

            flat_grad_tensor = torch.cat(flat_grads)

            # --- 2. Update Adam Moments (Flattened) ---
            exp_avg = group_state['exp_avg']
            exp_avg_sq = group_state['exp_avg_sq']

            exp_avg.mul_(beta1).add_(flat_grad_tensor, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(flat_grad_tensor, flat_grad_tensor, value=1.0 - beta2)

            # Bias corrected moments
            m_hat = exp_avg / bias_correction1
            d_hat = exp_avg_sq / bias_correction2 # Diagonal variance estimate

            # --- 3. Calculate Diagonal Preconditioner ---
            # s = sqrt(d_hat) + eps
            diag_precond = d_hat.sqrt().add_(eps)

            # --- 4. Update Householder Vectors (if k > 0) ---
            if k > 0:
                # Candidate vector for new reflection (diagonally preconditioned momentum)
                candidate_v = m_hat / diag_precond

                # Store in circular buffer
                current_index = group_state['vector_index']

                # Normalize? The reflection formula H = I - 2*v*v^T / (v^T v) handles scale,
                # but normalizing might improve stability if norms vary wildly.
                # Let's store unnormalized but use safe division.
                v_norm_sq = torch.dot(candidate_v, candidate_v)

                # Avoid storing zero vectors or vectors causing NaN/Inf
                if torch.isfinite(v_norm_sq) and v_norm_sq > 1e-12: # Add a threshold
                    group_state['householder_vectors'][current_index] = candidate_v
                    group_state['vector_norms_sq'][current_index] = v_norm_sq
                    group_state['vector_index'] = (current_index + 1) % k
                else:
                    # Handle potential instability: skip update or store None?
                    # Storing None ensures it's skipped during application.
                    group_state['householder_vectors'][current_index] = None
                    group_state['vector_norms_sq'][current_index] = 0.0
                    # Don't advance index if storing None? Or advance and leave None?
                    # Let's advance, simpler logic. The application loop will skip None.
                    group_state['vector_index'] = (current_index + 1) % k


            # --- 5. Compute Update Direction ---
            # Start with bias-corrected momentum
            update_vec = m_hat.clone() # Make a copy to modify

            # Apply Householder reflections (if k > 0)
            # Iterate through stored vectors. Order matters depending on the matrix
            # represented. Applying newest-to-oldest corresponds to P = H_k ... H_1 D^-1/2
            # Let's iterate through the buffer logically from newest to oldest insertion.
            if k > 0:
                idx = group_state['vector_index'] # next slot to be filled
                for i in range(k):
                    # Get index of vector to apply (newest is index-1, oldest is index)
                    vector_idx = (idx - 1 - i + k) % k
                    v = group_state['householder_vectors'][vector_idx]
                    v_norm_sq = group_state['vector_norms_sq'][vector_idx]

                    if v is not None and v_norm_sq > 1e-12: # Check if valid vector
                        # Apply H = I - 2*v*v^T / v_norm_sq
                        # update_vec = H @ update_vec
                        #            = update_vec - 2 * v * dot(v, update_vec) / v_norm_sq
                        v_dot_update = torch.dot(v, update_vec)

                        # --- CORRECTED LINE ---
                        # Use add_ for vector + alpha * vector
                        update_vec.add_(v, alpha=(-2.0 * v_dot_update / v_norm_sq))
                        # --- END CORRECTION ---

            # Apply diagonal preconditioner
            update_vec.div_(diag_precond)

            # Apply learning rate (negative sign for gradient descent)
            update_vec.mul_(-lr)

            # --- 6. Unflatten Update and Apply to Parameters ---
            offset = 0
            for p in group_params:
                numel = p.numel()
                p.add_(update_vec[offset : offset + numel].reshape(p.shape))
                offset += numel

            # Check if offset matches total size (sanity check)
            if offset != group_state['total_param_size']:
                 warnings.warn(f"Parameter size mismatch in group {id(group)}: calculated {offset}, expected {group_state['total_param_size']}")


        return loss