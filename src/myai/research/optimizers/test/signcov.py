import torch
from torch.optim.optimizer import Optimizer

class SignCov(Optimizer):
    def __init__(self, params, k=10, beta=0.9, lr_cov=1e-4, lambda_=1e-3, lr=1e-3):
        defaults = dict(k=k, beta=beta, lr_cov=lr_cov, lambda_=lambda_, lr=lr)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                d = p.numel()
                k = group['k']
                # Initialize with smaller values for stability
                state['U'] = torch.randn(d, k, device=p.device) * (1e-4 / (k**0.5))
                state['V'] = torch.randn(d, k, device=p.device) * (1e-4 / (k**0.5))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta = group['beta']
            lr_cov = group['lr_cov']
            lambda_ = group['lambda_']
            lr_param = group['lr']
            k = group['k']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                U = state['U']
                V = state['V']
                d = p.numel()

                # Flatten parameter and gradient
                p_flat = p.data.view(-1)
                grad_flat = grad.view(-1)
                s = torch.sign(grad_flat).float()

                # Update U and V with gradient clipping
                with torch.no_grad():
                    U_prev, V_prev = U.clone(), V.clone()

                    # Update U
                    VtV = V_prev.t() @ V_prev
                    term1_U = U @ VtV
                    term2_U = beta * (U_prev @ (V_prev.t() @ V))
                    term3_U = (1 - beta) * s.unsqueeze(-1) * (s @ V_prev)
                    U -= lr_cov * (term1_U - term2_U - term3_U)

                    # Update V
                    UtU = U_prev.t() @ U_prev
                    term1_V = V @ UtU
                    term2_V = beta * (V_prev @ (U_prev.t() @ U))
                    term3_V = (1 - beta) * s.unsqueeze(-1) * (s @ U_prev)
                    V -= lr_cov * (term1_V - term2_V - term3_V)

                # Precondition gradient using Woodbury identity
                VtU = V.t() @ U
                I = torch.eye(k, device=p.device)
                matrix = I + (1.0 / lambda_) * VtU
                matrix = 0.5 * (matrix + matrix.T)  # Enforce symmetry

                # Adaptive jitter for positive definiteness
                jitter = 1e-6
                max_jitter_attempts = 5
                for _ in range(max_jitter_attempts):
                    try:
                        L = torch.linalg.cholesky(matrix)
                        break
                    except torch._C._LinAlgError:
                        matrix += I * jitter
                        jitter *= 10
                else:
                    raise RuntimeError("Matrix not positive definite after jitter.")

                Vt_grad = V.t() @ grad_flat
                rhs = Vt_grad.unsqueeze(-1)
                sol = torch.cholesky_solve(rhs, L).squeeze(-1)

                preconditioned_grad = (1.0 / lambda_) * grad_flat - (1.0 / (lambda_**2)) * (U @ sol)
                p_flat -= lr_param * preconditioned_grad
                p.data.copy_(p_flat.view_as(p.data))

        return loss