import torch
import torch.nn.functional as F

class NavierStokesMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, viscosity=1.0, diffusion=0.5):
        """
        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float, optional): Learning rate (time step). Default: 1e-3.
            viscosity (float, optional): mu - viscosity coefficient (damping). Default: 0.1.
            diffusion (float, optional): nu - diffusion coefficient. Default: 0.01.
        """
        defaults = dict(lr=lr, mu=viscosity, nu=diffusion)
        super().__init__(params, defaults)

        # Initialize velocity buffers
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['velocity'] = torch.zeros_like(p)

    def compute_laplacian(self, v):
        laplacian = torch.zeros_like(v)

        for dim in range(v.ndim):
            # Permute target dimension to end
            perm = list(range(v.ndim))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            v_p = v.permute(perm)

            # Manually pad with edge values
            padded = torch.cat([
                v_p[..., :1],  # First element
                v_p,
                v_p[..., -1:], # Last element
            ], dim=-1)

            # Compute second difference
            diff = padded[..., 2:] - 2 * padded[..., 1:-1] + padded[..., :-2]

            # Permute back and accumulate
            laplacian += diff.permute(perm)

        return laplacian

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            nu = group['nu']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                velocity = self.state[p]['velocity']

                # Compute Laplacian of velocity
                laplacian_v = self.compute_laplacian(velocity)

                # Navier-Stokes momentum
                velocity_update = lr * (-grad + nu * laplacian_v - mu * velocity)
                velocity.add_(velocity_update)

                # Update parameters using velocity
                p.data.add_(lr * velocity)

        return loss