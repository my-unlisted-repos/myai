import math
from collections import deque

import torch
from torch.optim import Optimizer


class SubspaceGP(Optimizer):
    """gaussian processes in random subspace - guided updates"""
    def __init__(self, params, d=20, hist_size=10, lr=0.01, length_scale=1.0, signal_var=1.0, noise_var=0.1):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        # Gather all parameters to determine total dimensions
        params_list = []
        for group in self.param_groups:
            params_list.extend(group['params'])
        with torch.no_grad():
            flat_params = torch.cat([p.view(-1).detach() for p in params_list])

        self.d = d
        self.hist_size = hist_size
        self.total_params = flat_params.numel()
        self.lr = lr
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise_var = noise_var

        # Initialize projection matrix R (d x total_params)
        self.R = torch.randn(d, self.total_params, device=flat_params.device) / math.sqrt(d)

        # History storage for (z, loss)
        self.history = deque(maxlen=hist_size)

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for GaussianProcessOptimizer")

        # Compute loss and gradients if necessary
        loss = closure()

        # Gather all parameters into a flat tensor
        params_list = []
        for group in self.param_groups:
            params_list.extend(group['params'])
        with torch.no_grad():
            flat_params = torch.cat([p.view(-1).detach() for p in params_list])

        # Project current parameters to z
        z_current = torch.matmul(self.R, flat_params)

        # Store current z and loss in history
        self.history.append((z_current.detach().clone(), loss.item()))

        # If history is too small, perform SGD step
        if len(self.history) < 2:
            for p in params_list:
                if p.grad is not None:
                    p.data.add_(-self.lr, p.grad.data)
            return loss

        # Prepare data for GP
        Z = torch.stack([h[0] for h in self.history])  # (hist_size, d)
        y = torch.tensor([h[1] for h in self.history], device=Z.device, dtype=torch.float32)

        # Compute RBF kernel matrix
        K = self.rbf_kernel(Z, Z)
        K += torch.eye(K.size(0), device=K.device) * self.noise_var

        # Solve for alpha = K^{-1}y
        try:
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)
        except RuntimeError:
            alpha = torch.linalg.pinv(K) @ y

        # Compute kernel vector for current z
        k_star = self.rbf_kernel(z_current.unsqueeze(0), Z).squeeze(0)

        # Compute gradient of the mean function at z_current
        diff = Z - z_current.unsqueeze(0)
        weights = alpha * k_star
        gradient_z = (weights.unsqueeze(-1) * diff).sum(dim=0) / (self.length_scale ** 2)

        # Project gradient back to parameter space
        update_flat = torch.matmul(self.R.T, gradient_z)

        # Distribute update to parameters
        idx = 0
        for p in params_list:
            p_size = p.numel()
            p_update = update_flat[idx: idx + p_size].view_as(p)
            p.data.add_(-self.lr, p_update)
            idx += p_size

        return loss

    def rbf_kernel(self, X, Y):
        # X: (n, d), Y: (m, d)
        sq_dist = torch.cdist(X, Y, p=2)**2
        return self.signal_var * torch.exp(-sq_dist / (2 * self.length_scale**2))