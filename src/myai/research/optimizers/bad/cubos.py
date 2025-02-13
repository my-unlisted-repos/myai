# pylint:disable=signature-differs, not-callable # type:ignore

import torch
from torch.optim import Optimizer

class CUBOS(Optimizer):
    """Cubic Zeroth-Order Stochastic Optimizer"""
    def __init__(self, params, lr=1, h=1e-2):
        defaults = dict(lr=lr, h=h)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure):
        if closure is None:
            raise ValueError("CUBOS requires a closure to compute the loss.")

        # Store original parameters and compute initial loss
        orig_loss = closure(False)
        orig_params = []
        for group in self.param_groups:
            for p in group['params']:
                orig_params.append(p.detach().clone())

        # Generate a random direction vector
        vs = []
        for group in self.param_groups:
            for p in group['params']:
                v = torch.randn_like(p)
                vs.append(v)
        # Calculate norm across all parameters
        total_norm = torch.sqrt(sum(torch.sum(v**2) for v in vs))
        if total_norm == 0:
            return orig_loss
        vs = [v / total_norm for v in vs]

        # Evaluate function at perturbations
        f = []
        deltas = [0, group['h'], -group['h'], 2*group['h'], -2*group['h']]
        for delta in deltas:
            # Apply perturbation
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    p.add_(delta * vs[idx])
                    idx += 1
            # Compute loss
            loss = closure(False)
            f.append(loss.detach().item())
            # Restore parameters
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    p.copy_(orig_params[idx])
                    idx += 1

        # Fit cubic model: f(x + δ) = aδ³ + bδ² + cδ + d (d = f[0])
        # Using deltas h, -h, 2h, -2h (skip δ=0)
        M = []
        targets = []
        h = self.param_groups[0]['h']
        for delta in deltas[1:]:
            row = [delta**3, delta**2, delta]
            M.append(row)
            targets.append(f[1 + deltas[1:].index(delta)] - f[0])
        M_tensor = torch.tensor(M, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        # Solve least squares
        try:
            coeffs = torch.linalg.lstsq(M_tensor, targets_tensor).solution
        except:
            coeffs = torch.zeros(3)
        a, b, c = coeffs[0], coeffs[1], coeffs[2]

        # Compute optimal delta
        discriminant = (2*b)**2 - 4*3*a*c
        if discriminant < 0:
            delta_opt = 0.0
        else:
            sqrt_disc = torch.sqrt(discriminant)
            delta1 = (-2*b + sqrt_disc) / (6*a + 1e-8)
            delta2 = (-2*b - sqrt_disc) / (6*a + 1e-8)
            # Check second derivative for minima
            sec_deriv1 = 6*a*delta1 + 2*b
            sec_deriv2 = 6*a*delta2 + 2*b
            valid = []
            if sec_deriv1 > 0:
                valid.append(delta1)
            if sec_deriv2 > 0:
                valid.append(delta2)
            if valid:
                f_vals = [a*d**3 + b*d**2 + c*d + f[0] for d in valid]
                delta_opt = valid[torch.argmin(torch.tensor(f_vals))]
            else:
                delta_opt = 0.0

        # Apply update
        lr = self.param_groups[0]['lr']
        delta_step = delta_opt * lr
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                p.add_(delta_step * vs[idx])
                idx += 1

        return torch.tensor(f[0], dtype=orig_loss.dtype, device=orig_loss.device)