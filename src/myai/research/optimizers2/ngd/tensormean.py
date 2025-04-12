import math
import warnings
from typing import List, Optional, Tuple

import torch
from torch import optim
from torch import Tensor


# Helper function for Matrix Inverse Square Root using Eigenvalue Decomposition
def matrix_inverse_sqrt(matrix: Tensor, epsilon: float = 1e-10) -> Tensor:
    """
    Computes the inverse square root of a positive semi-definite matrix.
    M^(-1/2) = V @ diag(1 / sqrt(D + epsilon)) @ V.T
    Args:
        matrix: The matrix to compute the inverse square root of. Must be symmetric.
        epsilon: Small value added to eigenvalues for numerical stability.
    Returns:
        The inverse square root of the matrix.
    """
    # Ensure matrix is on the correct device and dtype
    device = matrix.device
    dtype = matrix.dtype

    # Add epsilon * I for stability before decomposition
    # Create identity matrix on the same device and dtype
    identity = torch.eye(matrix.size(0), device=device, dtype=dtype)
    matrix_stable = matrix + identity * epsilon

    try:
        # Eigenvalue decomposition: matrix = V @ diag(D) @ V.T
        # eigh requires input to be symmetric/Hermitian. Ensure this or use svd.
        # Let's assume gradients lead to symmetric outer products E[gg^T]
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix_stable)

        # Clamp eigenvalues to be non-negative before sqrt
        eigenvalues = torch.clamp(eigenvalues, min=epsilon)

        # Compute 1 / sqrt(eigenvalues)
        inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues) # No need to add epsilon again if added before

        # Reconstruct the matrix inverse square root
        # M^(-1/2) = V @ diag(inv_sqrt_eigenvalues) @ V.T
        # Note: V returned by eigh has columns as eigenvectors. V.T is correct here.
        inv_sqrt_matrix = eigenvectors @ torch.diag(inv_sqrt_eigenvalues) @ eigenvectors.T

    except torch._C._LinAlgError as e:
        warnings.warn(f"Warning: linalg.eigh failed: {e}. Falling back to adding larger epsilon to diagonal.")
        # Fallback: Add more epsilon and try again or return identity/scaled identity
        try:
            # Try adding more epsilon to the original matrix diagonal
            matrix_stable = matrix + torch.eye(matrix.size(0), device=device, dtype=dtype) * epsilon * 100
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix_stable)
            eigenvalues = torch.clamp(eigenvalues, min=1e-8) # Prevent division by zero more strictly
            inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues)
            inv_sqrt_matrix = eigenvectors @ torch.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
        except Exception as fallback_e:
            warnings.warn(f"Error: Fallback failed: {fallback_e}. Returning identity.")
            inv_sqrt_matrix = torch.eye(matrix.size(0), device=device, dtype=dtype)


    return inv_sqrt_matrix


class AdamMeanCov(optim.Optimizer):
    """
    Implements Adam-like algorithm using full covariance of mean gradients.

    For N-D tensors (N>1), it computes means along N-1 dimensions N times
    and maintains covariance matrices for these mean vectors.
    For 1D tensors, it maintains a full covariance matrix for the tensor itself.
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0):
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the diagonal of cov matrices
                 or to eigenvalues for numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
             raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

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
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamMeanCov does not support sparse gradients')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Store lists of means and cov matrices
                    state['exp_avg_means'] = [] # List[Tensor] - First moments (m_i)
                    state['exp_avg_sq_means'] = [] # List[Tensor] - Second raw moments (M2_i)

                    if p.dim() == 0: # Scalar parameter
                        state['exp_avg_means'].append(torch.zeros_like(p, memory_format=torch.preserve_format))
                        state['exp_avg_sq_means'].append(torch.zeros((1,1), dtype=p.dtype, device=p.device)) # scalar M2 is 1x1
                    elif p.dim() == 1: # 1D Tensor
                        dim_size = p.shape[0]
                        state['exp_avg_means'].append(torch.zeros_like(p, memory_format=torch.preserve_format))
                        state['exp_avg_sq_means'].append(torch.zeros((dim_size, dim_size), dtype=p.dtype, device=p.device))
                    else: # N-D Tensor (N > 1)
                        for i in range(p.dim()):
                            dim_size = p.shape[i]
                            state['exp_avg_means'].append(torch.zeros(dim_size, dtype=p.dtype, device=p.device))
                            state['exp_avg_sq_means'].append(torch.zeros((dim_size, dim_size), dtype=p.dtype, device=p.device))

                state['step'] += 1
                step = state['step']
                exp_avg_means = state['exp_avg_means']
                exp_avg_sq_means = state['exp_avg_sq_means']

                # Apply weight decay (L2 penalty)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # --- Calculate mean gradients and update moments ---
                combined_update = torch.zeros_like(p, memory_format=torch.preserve_format)
                num_dims_processed = 0

                if p.dim() == 0: # Scalar case
                    g_mean_0 = grad.unsqueeze(0) # Treat as 1D vector of size 1
                    m0 = exp_avg_means[0]
                    M2_0 = exp_avg_sq_means[0]

                    m0.mul_(beta1).add_(g_mean_0, alpha=1 - beta1)
                    outer_prod = g_mean_0.unsqueeze(-1) @ g_mean_0.unsqueeze(-2) # Shape (1, 1)
                    M2_0.mul_(beta2).add_(outer_prod, alpha=1 - beta2)

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    m0_hat = m0 / bias_correction1
                    M2_0_hat = M2_0 / bias_correction2

                    # Precondition
                    precond_0 = matrix_inverse_sqrt(M2_0_hat, eps) # Shape (1, 1)
                    delta_0 = precond_0 @ m0_hat.unsqueeze(-1) # Shape (1, 1)
                    combined_update.add_(delta_0.squeeze()) # Add the scalar update
                    num_dims_processed = 1


                elif p.dim() == 1: # 1D case
                    dim_size = p.shape[0]
                    g_mean_0 = grad # The "mean" is the gradient itself
                    m0 = exp_avg_means[0]
                    M2_0 = exp_avg_sq_means[0]

                    # Update moments
                    m0.mul_(beta1).add_(g_mean_0, alpha=1 - beta1)
                    # Outer product: treat g_mean_0 as column vector
                    outer_prod = g_mean_0.unsqueeze(-1) @ g_mean_0.unsqueeze(-2) # Shape (dim_size, dim_size)
                    M2_0.mul_(beta2).add_(outer_prod, alpha=1 - beta2)

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    m0_hat = m0 / bias_correction1
                    M2_0_hat = M2_0 / bias_correction2

                    # Precondition
                    precond_0 = matrix_inverse_sqrt(M2_0_hat, eps) # Shape (dim_size, dim_size)
                    # Treat m0_hat as column vector for matmul
                    delta_0 = precond_0 @ m0_hat.unsqueeze(-1) # Shape (dim_size, 1)

                    combined_update.add_(delta_0.squeeze(-1)) # Shape (dim_size,)
                    num_dims_processed = 1

                else: # N-D case (N > 1)
                    tensor_dims = tuple(range(p.dim()))
                    for i in range(p.dim()):
                        dims_to_reduce = tuple(d for d in tensor_dims if d != i)
                        g_mean_i = torch.mean(grad, dim=dims_to_reduce) # Shape (di,)

                        mi = exp_avg_means[i]
                        M2_i = exp_avg_sq_means[i]

                        # Update moments
                        mi.mul_(beta1).add_(g_mean_i, alpha=1 - beta1)
                        outer_prod = g_mean_i.unsqueeze(-1) @ g_mean_i.unsqueeze(-2) # Shape (di, di)
                        M2_i.mul_(beta2).add_(outer_prod, alpha=1 - beta2)

                        # Bias correction
                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step
                        mi_hat = mi / bias_correction1
                        M2_i_hat = M2_i / bias_correction2

                        # Precondition
                        precond_i = matrix_inverse_sqrt(M2_i_hat, eps) # Shape (di, di)
                        delta_i = precond_i @ mi_hat.unsqueeze(-1) # Shape (di, 1)
                        delta_i = delta_i.squeeze(-1) # Shape (di,)

                        # Broadcast delta_i and add to combined update
                        # Reshape delta_i to align with the i-th dimension
                        broadcast_shape = [1] * p.dim()
                        broadcast_shape[i] = p.shape[i]
                        delta_i_broadcast = delta_i.view(broadcast_shape)

                        combined_update.add_(delta_i_broadcast)
                        num_dims_processed += 1

                # Average the updates from different dimensions if N-D
                if num_dims_processed > 1:
                    combined_update.div_(num_dims_processed) # Average effect

                # --- Apply update ---
                p.add_(combined_update, alpha=-lr)

        return loss