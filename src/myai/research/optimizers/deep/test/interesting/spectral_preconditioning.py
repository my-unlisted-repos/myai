# pylint: disable = not-callable, signature-differs
import torch
from torch import nn
from torch.optim import Optimizer


class StochasticSpectralPreconditioning(Optimizer):
    def __init__(self, params, lr=0.1, tau=2, alpha=0.1, eta=0.01, ):
        defaults = dict(tau=tau, alpha=alpha, eta=eta, lr=lr)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                n = p.numel()
                tau_group = group['tau']
                # Initialize eigenvectors using random orthogonal basis
                V = torch.randn(n, tau_group, device=p.device)
                V, _ = torch.linalg.qr(V)
                state['V'] = V  # shape [n, tau]
                state['a'] = torch.zeros(tau_group, device=p.device)  # eigenvalues

    def step(self, closure):
        if closure is None:
            raise ValueError("Closure required for gradient computation")

        loss = closure()

        for group in self.param_groups:
            alpha = group['alpha']
            tau = group['tau']
            eta = group['eta']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                V = state['V']
                a = state['a']
                g = p.grad.data
                shape = p.shape
                g_flat = g.view(-1).detach()
                tau = min(tau, g_flat.numel())

                # Online covariance approximation and power iteration
                with torch.no_grad():
                    # Normalize gradient for stability
                    g_norm = g_flat.norm()
                    if g_norm > 0:
                        g_hat = g_flat / g_norm

                        # Oja's rule update for online PCA
                        for i in range(tau):
                            v = V[:, i]
                            # Update eigenvector estimate
                            v_update = eta * (g_hat * torch.dot(g_hat, v) - v * torch.sum(v * g_hat * g_hat))
                            V[:, i] += v_update

                        # Orthogonalize using modified Gram-Schmidt
                        for i in range(tau):
                            v = V[:, i]
                            for j in range(i):
                                u = V[:, j]
                                v -= torch.dot(u, v) * u
                            V[:, i] = v / v.norm()

                        # Update eigenvalue estimates (moving average)
                        a_update = torch.matmul(V.t(), g_flat).pow(2)
                        a = 0.9 * a + 0.1 * a_update

                # Compute preconditioned gradient using Sherman-Morrison
                with torch.no_grad():
                    Vt_g = torch.mv(V.t(), g_flat)
                    diag = a / (alpha + a)
                    V_term = torch.mv(V, diag * Vt_g)
                    preconditioned_g_flat = (g_flat - V_term) / alpha
                    preconditioned_g = preconditioned_g_flat.view(shape)

                # Update parameters
                p.data.add_(-lr * preconditioned_g)

        return loss