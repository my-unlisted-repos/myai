import torch
from torch.optim import Optimizer

class ProcrustesOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('ProcrustesOptimizer does not support sparse gradients')

                # Reshape parameter and gradient into 2D matrix
                original_shape = p.shape
                if p.dim() > 2:
                    # Handle higher-dimensional tensors (e.g., Conv layers)
                    p_matrix = p.view(original_shape[0], -1)
                    grad_matrix = grad.view(original_shape[0], -1)
                elif p.dim() == 1:
                    p_matrix = p.unsqueeze(0)
                    grad_matrix = grad.unsqueeze(0)
                else:
                    p_matrix = p.clone(memory_format=torch.contiguous_format)
                    grad_matrix = grad.clone(memory_format=torch.contiguous_format)

                # Compute SGD step
                p_sgd_matrix = p_matrix - lr * grad_matrix

                # Check if parameter is near-zero to avoid division by zero
                norm_p = torch.norm(p_matrix)
                if norm_p < eps:
                    updated_p_matrix = p_sgd_matrix
                else:
                    # Compute M = p_matrix^T @ p_sgd_matrix
                    M = torch.matmul(p_matrix.T, p_sgd_matrix)

                    # Add small regularization to M for numerical stability
                    M_reg = M + eps * torch.eye(M.size(0), device=M.device, dtype=M.dtype)

                    # Compute SVD of M_reg
                    try:
                        U, S, Vh = torch.linalg.svd(M_reg, full_matrices=False)
                    except:
                        # Fallback to SGD if SVD fails
                        updated_p_matrix = p_sgd_matrix
                    else:
                        V = Vh.mH
                        # Compute orthogonal matrix Q = U @ V^T
                        Q = torch.matmul(U, V.T)

                        # Compute scaling factor s
                        norm_p_sq = torch.sum(p_matrix ** 2) + eps
                        s = torch.sum(S) / norm_p_sq

                        # Apply Procrustes transformation
                        updated_p_matrix = s * torch.matmul(p_matrix, Q)

                # Reshape back to original dimensions
                updated_p = updated_p_matrix.view(original_shape)

                # Update the parameter
                p.copy_(updated_p)

        return loss