import torch
from torch.optim import Optimizer
import math

class SNGE(Optimizer):
    def __init__(self, params, lr=0.01, sigma=0.1, eta_mu=0.1, eta_sigma=0.1):
        defaults = dict(lr=lr, sigma=sigma, eta_mu=eta_mu, eta_sigma=eta_sigma)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sigma'] = torch.full_like(p, group['sigma'])

    @torch.no_grad
    def step(self, closure):
        with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            eta_mu = group['eta_mu']
            eta_sigma = group['eta_sigma']
            lr = group['lr']

            # Collect parameters and states
            params = []
            states = []
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                states.append(self.state[p])

            # Antithetic perturbations
            epsilon = []
            for p, state in zip(params, states):
                eps = torch.normal(mean=0, std=1, size=p.shape, device=p.device)
                eps.mul_(state['sigma'])
                epsilon.append(eps)

            # Compute theta+ and theta-
            theta_plus = []
            theta_minus = []
            for p, eps in zip(params, epsilon):
                theta_plus.append(p.detach() + eps)
                theta_minus.append(p.detach() - eps)

            # Evaluate losses
            with torch.no_grad():
                original_params = [p.detach().clone() for p in params]

                # Compute loss for theta+
                for p, t_p in zip(params, theta_plus):
                    p.copy_(t_p)
                loss_plus = closure(False)

                # Compute loss for theta-
                for p, t_m in zip(params, theta_minus):
                    p.copy_(t_m)
                loss_minus = closure(False)

                # Restore original parameters
                for p, orig in zip(params, original_params):
                    p.copy_(orig)

            # Assign fitness (negative loss)
            f_plus = -loss_plus
            f_minus = -loss_minus

            # Determine utilities (rank-based)
            if f_plus > f_minus:
                u_plus = 0.5
                u_minus = -0.5
            else:
                u_plus = -0.5
                u_minus = 0.5

            # Update parameters (mean)
            for p, state, eps in zip(params, states, epsilon):
                grad = p.grad.data
                if grad is None:
                    continue
                # Gradient step
                p.data.add_(grad, alpha=-lr)
                # xNES mean step from antithetic samples
                delta_mu = eta_mu * (u_plus * eps + u_minus * (-eps))
                p.data.add_(delta_mu)

            # Update sigma (covariance)
            for state, eps in zip(states, epsilon):
                z_plus = eps / state['sigma']
                z_minus = -eps / state['sigma']

                # Compute delta_sigma
                term_plus = u_plus * (z_plus**2 - 1)
                term_minus = u_minus * (z_minus**2 - 1)
                delta_sigma = eta_sigma * state['sigma'] * (term_plus + term_minus) / 2
                state['sigma'].mul_(torch.exp(delta_sigma))

        return loss