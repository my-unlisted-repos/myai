import math

import torch
import torch.optim as optim
from torch import Tensor


class AHLRD(optim.Optimizer):
    """
    Adaptive Hybrid Low-Rank Plus Diagonal (AHLRD) optimizer.

    This optimizer implements a model-agnostic approximation to natural gradient
    without using per-sample gradients in a batch. It combines a diagonal preconditioner
    with a low-rank approximation of the curvature matrix, and adaptively determines
    the optimal rank.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): coefficient for computing running average of squared gradients
            for diagonal component (default: 0.9)
        beta2 (float, optional): coefficient for computing running average of outer products
            for low-rank component (default: 0.999)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        damping (float, optional): damping factor for the preconditioner (default: 1e-4)
        update_freq (int, optional): frequency of low-rank component updates (default: 20)
        min_rank (int, optional): minimum rank for low-rank component (default: 5)
        max_rank (int, optional): maximum rank for low-rank component (default: 50)
        threshold (float, optional): eigenvalue threshold for adaptive rank selection (default: 0.9)
        warmup_steps (int, optional): number of steps to use diagonal-only preconditioner (default: 100)
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0, damping=1e-4, update_freq=20, min_rank=5,
                 max_rank=50, threshold=0.9, warmup_steps=100):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, damping=damping,
                        update_freq=update_freq, min_rank=min_rank,
                        max_rank=max_rank, threshold=threshold)

        super(AHLRD, self).__init__(params, defaults)

        self.warmup_steps = warmup_steps
        self._steps = 0

        # Initialize state for each parameter group
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Initialize low-rank component state
                param_shape = p.shape
                param_dim = p.numel()

                # For vector parameters, we need special handling
                if len(param_shape) == 1:
                    state['is_vector'] = True
                    # For vectors, we'll use the full rank (which is just 1)
                    state['rank'] = 1
                    state['cov_estimate'] = torch.zeros((param_dim, param_dim),
                                                       device=p.device, dtype=p.dtype)
                    state['low_rank_U'] = torch.zeros((param_dim, 1),
                                                     device=p.device, dtype=p.dtype)
                    state['low_rank_S'] = torch.zeros(1, device=p.device, dtype=p.dtype)
                else:
                    state['is_vector'] = False
                    state['rank'] = group['min_rank']
                    state['cov_estimate'] = torch.zeros((param_dim, param_dim),
                                                       device=p.device, dtype=p.dtype)
                    state['low_rank_U'] = torch.zeros((param_dim, group['min_rank']),
                                                     device=p.device, dtype=p.dtype)
                    state['low_rank_S'] = torch.zeros(group['min_rank'],
                                                     device=p.device, dtype=p.dtype)

    def _update_low_rank_component(self, p, group, state):
        """Update the low-rank component of the preconditioner."""
        param_dim = p.numel()

        # Skip low-rank updates for very small parameters or during warmup
        if param_dim <= 1 or self._steps < self.warmup_steps:
            return

        # Compute eigendecomposition of covariance estimate
        try:
            # Use torch.linalg.eigh for better numerical stability
            S, U = torch.linalg.eigh(state['cov_estimate'])

            # Sort eigenvalues and eigenvectors in descending order
            S = S.flip(0)
            U = U.flip(1)

            # Ensure eigenvalues are positive
            S = torch.clamp(S, min=0.0)

            # Adaptive rank selection based on eigenvalue threshold
            if not state['is_vector']:
                # Compute normalized cumulative sum
                total_variance = S.sum().item()
                if total_variance > 0:
                    cumulative_variance = torch.cumsum(S, dim=0) / total_variance

                    # Find smallest r such that cumulative_variance[r-1] >= threshold
                    r = torch.searchsorted(cumulative_variance, group['threshold']) + 1
                    r = min(max(r.item(), group['min_rank']), min(group['max_rank'], param_dim))

                    # Update rank
                    state['rank'] = int(r)

            # Truncate to current rank
            r = state['rank']
            state['low_rank_U'] = U[:, :r]
            state['low_rank_S'] = S[:r]

        except RuntimeError as e:
            # Fallback to diagonal preconditioner if eigendecomposition fails
            print(f"Warning: Eigendecomposition failed with error: {e}. Using diagonal preconditioner.")
            state['low_rank_U'] = torch.zeros((param_dim, 1), device=p.device, dtype=p.dtype)
            state['low_rank_S'] = torch.zeros(1, device=p.device, dtype=p.dtype)
            state['rank'] = 1

    def _compute_preconditioned_gradient(self, p, grad, group, state):
        """Compute the preconditioned gradient using the hybrid preconditioner."""
        # Get diagonal component (D)
        D = state['exp_avg_sq'] + group['damping']
        D_sqrt_inv = 1.0 / torch.sqrt(D)

        # During warmup, use only diagonal preconditioner
        if self._steps < self.warmup_steps or state['rank'] == 0:
            return grad / D

        # Reshape parameters and gradients to vectors for matrix operations
        orig_shape = p.shape
        grad_flat = grad.view(-1)
        D_flat = D.view(-1)
        D_sqrt_inv_flat = D_sqrt_inv.view(-1)

        # Get low-rank component (L)
        U = state['low_rank_U']
        S = state['low_rank_S']

        # Scale U by D^(1/2)
        L = U * D_sqrt_inv_flat.unsqueeze(1) * torch.sqrt(S).unsqueeze(0)

        # Compute preconditioned gradient using Woodbury identity
        # P = D^(-1) - D^(-1) * L * (I + L^T * D^(-1) * L)^(-1) * L^T * D^(-1)

        # First compute D^(-1) * grad
        D_inv_grad = grad_flat / D_flat

        # Compute L^T * D^(-1) * grad
        LT_D_inv_grad = torch.matmul(L.t(), D_inv_grad)

        # Compute (I + L^T * D^(-1) * L)
        M = torch.eye(state['rank'], device=p.device, dtype=p.dtype)
        M = M + torch.matmul(L.t(), L)

        # Compute (I + L^T * D^(-1) * L)^(-1) * L^T * D^(-1) * grad
        try:
            M_inv_LT_D_inv_grad = torch.linalg.solve(M, LT_D_inv_grad)
        except RuntimeError:
            # Fallback if matrix inversion fails
            M = M + torch.eye(state['rank'], device=p.device, dtype=p.dtype) * group['eps']
            M_inv_LT_D_inv_grad = torch.linalg.solve(M, LT_D_inv_grad)

        # Compute D^(-1) * L * (I + L^T * D^(-1) * L)^(-1) * L^T * D^(-1) * grad
        correction = torch.matmul(L, M_inv_LT_D_inv_grad)

        # Final preconditioned gradient
        preconditioned_grad = D_inv_grad - correction

        # Reshape back to original shape
        return preconditioned_grad.view(orig_shape)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._steps += 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AHLRD does not support sparse gradients')

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]
                state['step'] += 1

                # Update diagonal component (similar to Adam)
                state['exp_avg_sq'].mul_(group['beta1']).addcmul_(grad, grad, value=1 - group['beta1'])

                # Compute preconditioned gradient for parameter update
                if self._steps < self.warmup_steps:
                    # During warmup, use only diagonal preconditioner
                    D = state['exp_avg_sq'] + group['damping']
                    preconditioned_grad = grad / D
                else:
                    # Compute preconditioned gradient with full hybrid preconditioner
                    preconditioned_grad = self._compute_preconditioned_gradient(p, grad, group, state)

                # Update parameters
                p.add_(preconditioned_grad, alpha=-group['lr'])

                # Update low-rank component periodically
                if not state['is_vector'] and state['step'] % group['update_freq'] == 0:
                    # Compute preconditioned gradient for covariance update
                    D_sqrt_inv = 1.0 / torch.sqrt(state['exp_avg_sq'] + group['damping'])
                    g_tilde = grad.view(-1) * D_sqrt_inv.view(-1)

                    # Update covariance estimate
                    g_tilde_outer = torch.outer(g_tilde, g_tilde)
                    state['cov_estimate'].mul_(group['beta2']).add_(g_tilde_outer, alpha=1 - group['beta2'])

                    # Update low-rank component
                    self._update_low_rank_component(p, group, state)

        return loss


