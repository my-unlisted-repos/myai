import torch
from torch.optim import Optimizer

class SubspaceQuadratic(Optimizer):
    """fits quadratic model in a random subspace."""
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for ExtremeSubspaceQuadraticOptimizer")

        # Initial loss and gradient computation
        loss = closure()
        # loss.backward()

        params_list = []
        grads = []
        rand_dirs = []
        lr = self.param_groups[0]['lr']

        # Prepare gradients and random directions for each parameter
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params_list.append(p)
                grads.append(p.grad.clone())
                # Generate and normalize random direction
                rand_dir = torch.randn_like(p)
                rand_dir_norm = torch.norm(rand_dir)
                if rand_dir_norm > 1e-8:
                    rand_dir = rand_dir / rand_dir_norm
                rand_dirs.append(rand_dir)

        # Save original parameters
        original_params = [p.data.clone() for p in params_list]

        # Define sample points in the subspace
        samples = [
            (0.0, 0.0),    # Center
            (1.0, 0.0),    # Along gradient
            (-1.0, 0.0),   # Opposite gradient
            (0.0, 1.0),    # Along random direction
            (0.0, -1.0),   # Opposite random direction
            (1.0, 1.0),    # Diagonal
            (-1.0, -1.0),  # Opposite diagonal
        ]
        losses = []

        # Evaluate loss for each sample
        for alpha, beta in samples:
            # Apply subspace perturbation
            for p, g, r in zip(params_list, grads, rand_dirs):
                p.data.copy_(original_params[params_list.index(p)] + lr * (alpha * g + beta * r))
            # Compute loss without gradients
            with torch.no_grad():
                current_loss = closure(False)
            losses.append(current_loss.item())
            # Restore original parameters
            for p, orig in zip(params_list, original_params):
                p.data.copy_(orig)

        # Fit quadratic model: loss = aα² + bβ² + cαβ + dα + eβ + f
        A = []
        for alpha, beta in samples:
            A.append([alpha**2, beta**2, alpha*beta, alpha, beta, 1.0])
        A_tensor = torch.tensor(A, dtype=torch.float32)
        losses_tensor = torch.tensor(losses, dtype=torch.float32)

        # Solve least squares for quadratic coefficients
        try:
            coeffs = torch.linalg.lstsq(A_tensor, losses_tensor).solution
        except RuntimeError:
            coeffs = torch.zeros(6)

        a, b, c, d, e, f = coeffs.tolist()

        # Compute optimal α and β by solving 2aα + cβ + d = 0 and cα + 2bβ + e = 0
        denominator = 4 * a * b - c**2
        if abs(denominator) > 1e-6:
            alpha_opt = (-2 * b * d + c * e) / denominator
            beta_opt = (-2 * a * e + c * d) / denominator
        else:
            # Fallback to best sample if model is not convex
            best_idx = torch.argmin(losses_tensor).item()
            alpha_opt, beta_opt = samples[best_idx]

        # Apply the optimal update
        for p, g, r in zip(params_list, grads, rand_dirs):
            update = lr * (alpha_opt * g + beta_opt * r)
            p.data.add_(update)

        # Clear gradients to avoid accumulation
        for p in params_list:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        return loss