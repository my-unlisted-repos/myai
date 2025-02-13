# pylint:disable = not-callable
import torch
from torch.optim.optimizer import Optimizer

class SubspaceInverse(Optimizer):
    """this is subpace based covariance matrix adaptation note k must be less than total number of variables"""
    def __init__(self, params, lr=1e-3, k=10, update_freq=100, m=20, beta=0.9, epsilon=1e-8):
        if k < 1:
            raise ValueError("Subspace dimension k must be at least 1")
        if m < k:
            raise ValueError("Gradient buffer size m must be at least k")
        defaults = dict(lr=lr, k=k, update_freq=update_freq, m=m, beta=beta, epsilon=epsilon)
        super().__init__(params, defaults)

        for group in self.param_groups:
            params = group['params']
            sizes = [p.numel() for p in params]
            total_params = sum(sizes)
            if total_params == 0:
                continue

            # Initialize orthonormal basis Q
            Q = torch.randn(total_params, k, device=params[0].device)
            Q, _ = torch.linalg.qr(Q)
            group['Q'] = Q
            # Initialize covariance matrix H
            group['H'] = torch.eye(k, device=params[0].device) * epsilon
            # Gradient buffer for basis updates
            group['grad_buffer'] = []
            group['step_count'] = 0
            # Save sizes for unflattening
            group['param_sizes'] = sizes

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if len(group['params']) == 0:
                continue

            lr = group['lr']
            k = group['k']
            update_freq = group['update_freq']
            m = group['m']
            beta = group['beta']
            epsilon = group['epsilon']
            Q = group['Q']
            H = group['H']
            grad_buffer = group['grad_buffer']
            step_count = group['step_count']
            param_sizes = group['param_sizes']

            # Flatten all gradients
            grads = []
            for p in group['params']:
                if p.grad is None:
                    grads.append(torch.zeros_like(p).view(-1))
                    continue
                grads.append(p.grad.detach().view(-1))
            g_flat = torch.cat(grads)

            # Store gradient in buffer
            if len(grad_buffer) < m:
                grad_buffer.append(g_flat.clone())
            else:
                grad_buffer[step_count % m] = g_flat.clone()

            # Update basis every update_freq steps
            if (step_count + 1) % update_freq == 0:
                # Collect gradients from buffer (may contain None if not filled)
                valid_grads = [g for g in grad_buffer if g is not None]
                if len(valid_grads) >= k:
                    G = torch.stack(valid_grads[:m], dim=1)
                    U, S, _ = torch.linalg.svd(G, full_matrices=False)
                    new_Q = U[:, :k]
                    new_Q, _ = torch.linalg.qr(new_Q)
                    group['Q'] = new_Q
                    Q = new_Q
                    # Reset covariance matrix
                    group['H'] = torch.eye(k, device=Q.device) * epsilon

            # Project gradient into subspace
            g_sub = torch.matmul(Q.t(), g_flat)

            # Update covariance matrix
            outer = torch.outer(g_sub, g_sub)
            group['H'] = beta * H + (1 - beta) * outer
            H = group['H']

            # Compute subspace update direction
            H_reg = H + epsilon * torch.eye(k, device=H.device)
            try:
                L = torch.linalg.cholesky(H_reg)
                Linv = torch.cholesky_inverse(L)
                delta_sub = torch.matmul(Linv, g_sub)
            except RuntimeError:
                # Fallback to diagonal if Cholesky fails
                diag_H = torch.diag(H_reg)
                delta_sub = g_sub / (diag_H + epsilon)

            # Project back to parameter space
            delta_flat = torch.matmul(Q, delta_sub)

            # Unflatten and apply updates
            idx = 0
            for p, size in zip(group['params'], param_sizes):
                if p.grad is None:
                    continue
                delta_p = delta_flat[idx:idx+size].view_as(p)
                p.data.add_(delta_p, alpha=-lr)
                idx += size

            group['step_count'] += 1

        return loss