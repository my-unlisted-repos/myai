import torch
import torch.nn as nn
from torch.optim import Optimizer

class KroneckerAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.dim() == 2:
                    m, n = p.shape
                    state['step'] = 0
                    state['M'] = torch.zeros_like(p)
                    state['A'] = torch.zeros(m, m, device=p.device)
                    state['B'] = torch.zeros(n, n, device=p.device)
                else:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

    @torch.no_grad
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if p.dim() == 2:
                    state['step'] += 1
                    step = state['step']
                    M, A, B = state['M'], state['A'], state['B']
                    m, n = p.shape

                    # Update first moment
                    M.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # Update Kronecker factors A and B
                    G = grad
                    delta_A = (G @ G.T) / n
                    A.mul_(beta2).add_(delta_A, alpha=1 - beta2)
                    delta_B = (G.T @ G) / m
                    B.mul_(beta2).add_(delta_B, alpha=1 - beta2)

                    # Bias correction
                    bc1 = 1 - beta1 ** step
                    bc2 = 1 - beta2 ** step
                    M_hat = M / bc1
                    A_hat = A / bc2 + eps * torch.eye(m, device=p.device)
                    B_hat = B / bc2 + eps * torch.eye(n, device=p.device)

                    # Compute inverse square roots via eigenvalue decomposition
                    def inv_sqrt(C, eps):
                        eigvals, eigvecs = torch.linalg.eigh(C)
                        eigvals = torch.clamp(eigvals, min=0) + eps  # Ensure positive
                        inv_sqrt_e = 1.0 / torch.sqrt(eigvals)
                        return eigvecs @ torch.diag(inv_sqrt_e) @ eigvecs.T

                    A_inv_sqrt = inv_sqrt(A_hat, eps)
                    B_inv_sqrt = inv_sqrt(B_hat, eps)

                    # Precondition the gradient
                    preconditioned_M = A_inv_sqrt @ M_hat @ B_inv_sqrt

                    # Update parameters
                    p.sub_(preconditioned_M, alpha=lr)
                else:
                    # Standard Adam for non-matrix parameters
                    state['step'] += 1
                    step = state['step']
                    m, v = state['m'], state['v']

                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bc1 = 1 - beta1 ** step
                    bc2 = 1 - beta2 ** step
                    m_hat = m / bc1
                    v_hat = v / bc2

                    denom = v_hat.sqrt().add_(eps)
                    p.addcdiv_(m_hat, denom, value=-lr)
        return loss