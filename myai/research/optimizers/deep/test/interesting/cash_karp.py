# pylint:disable=signature-differs, not-callable

import torch
from torch.optim import Optimizer

class CashKarp(Optimizer):
    """can Cash–Karp generate better directions? is it worth 6 evals?"""
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        # Cash-Karp coefficients
        self.a = [
            [1/5],
            [3/40, 9/40],
            [3/10, -9/10, 6/5],
            [-11/54, 5/2, -70/27, 35/27],
            [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]
        ]
        self.b5 = [37/378, 0, 250/621, 125/594, 0, 512/1771]
        self.c = [0, 1/5, 3/10, 3/5, 1, 7/8]

    @torch.no_grad
    def step(self, closure):
        if closure is None:
            raise ValueError("Must provide a closure for CashKarpOptimizer")

        group = self.param_groups[0]
        lr = group['lr']
        params = [p for g in self.param_groups for p in group['params']]

        # Filter parameters and save originals
        active_params = [p for p in params if p.requires_grad]
        original = [p.detach().clone() for p in active_params]
        device = original[0].device if len(original) > 0 else torch.device('cpu')
        n_params = len(active_params)
        ks = [[] for _ in range(n_params)]

        # Compute all 6 stages
        for stage in range(6):
            with torch.no_grad():
                for i, p in enumerate(active_params):
                    if stage == 0:  # First stage: k1 at original params
                        p.data.copy_(original[i])
                    else:  # Subsequent stages
                        if stage-1 >= len(self.a):
                            continue
                        a_coeffs = self.a[stage-1]
                        delta = sum(a_coeffs[j] * ks[i][j] for j in range(len(a_coeffs))) * lr
                        p.data.copy_(original[i] + delta)

            # Compute gradients for current stage
            with torch.enable_grad(): loss = closure()

            with torch.no_grad():
                for i, p in enumerate(active_params):
                    if p.grad is None:
                        ks[i].append(torch.zeros_like(p))
                    else:
                        ks[i].append(-p.grad.detach().clone())  # Negative gradient for ODE step
                self.zero_grad()

        # Restore original parameters and apply final update
        with torch.no_grad():
            for i, p in enumerate(active_params):
                if len(ks[i]) != 6:
                    raise RuntimeError("Incomplete stages calculation")

                # Calculate 5th-order update
                update = lr * sum(self.b5[s] * ks[i][s] for s in range(6))
                p.data.copy_(original[i] + update)

        return loss # type:ignore