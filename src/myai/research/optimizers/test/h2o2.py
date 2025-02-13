# pylint:disable=signature-differs, not-callable

import torch
from torch.optim import Optimizer

class H2O2(Optimizer):
    """
    Holistic Higher-Order Optimizer. Uses fourth-order Runge-Kutta update rule.
    Performs four gradient evaluations per step at strategically perturbed parameters to approximate higher-order terms.
    """

    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        # RK4 coefficients
        self.a = [0.0, 0.5, 0.5, 1.0]
        self.b = [1/6, 1/3, 1/3, 1/6]

        # Initialize previous parameters buffer
        self.prev_params = []
        for group in self.param_groups:
            self.prev_params.append([p.clone() for p in group['params']])

    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step using RK4-like gradient evaluations."""
        if closure is None:
            raise ValueError("Closure required for H2O2 (need to recompute loss multiple times)")

        # Store initial parameters and gradients
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                self.prev_params[i][j].copy_(p.data)

        k = []  # To store gradients at each stage
        # Compute k1 (standard gradient)
        self.zero_grad()
        with torch.enable_grad(): loss = closure()
        # loss.backward()
        k1 = []
        for group in self.param_groups:
            k1.append([p.grad.clone() for p in group['params']])
        k.append(k1)

        # Compute subsequent stages
        for stage in range(1, 4):
            # Restore initial parameters
            for i, group in enumerate(self.param_groups):
                for j, p in enumerate(group['params']):
                    p.data.copy_(self.prev_params[i][j])
                    if stage > 1:
                        # Apply previous perturbations
                        for s in range(stage):
                            a_coeff = self.a[s] * self.a[stage]
                            p.data.add_(k[s][i][j], alpha=-group['lr'] * a_coeff)

            # Compute gradient at perturbed position
            self.zero_grad()
            with torch.enable_grad(): loss = closure()
            k_stage = []
            for group in self.param_groups:
                k_stage.append([p.grad.clone() for p in group['params']])
            k.append(k_stage)

        # Combine gradients using RK4 coefficients
        for i, group in enumerate(self.param_groups):
            lr = group['lr']
            for j, p in enumerate(group['params']):
                total_update = torch.zeros_like(p)
                for s in range(4):
                    total_update.add_(k[s][i][j], alpha=self.b[s] * lr)
                p.data.sub_(total_update)

        return loss