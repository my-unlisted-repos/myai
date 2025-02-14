import torch
from torch.optim import Optimizer

class MetropolisHastings(Optimizer):
    """zo"""
    def __init__(self, params, sigma=0.01, T=1.0, temperature_decay=0.99, target_accept=0.234, adapt_sigma=True):
        defaults = dict(sigma=sigma, T=T, temperature_decay=temperature_decay,
                        target_accept=target_accept, adapt_sigma=adapt_sigma)
        super().__init__(params, defaults)

        self.state['step'] = 0
        self.state['accept_rate'] = 0.0

    @torch.no_grad
    def step(self, closure):
        if closure is None:
            raise ValueError("Closure required for MetropolisHastingsOptimizer")

        # Compute current loss and save parameters
        with torch.no_grad():
            current_loss = float(closure(False))

        # Save current parameters
        saved_params = [
            p.detach().clone() for group in self.param_groups for p in group['params']
        ]

        # Generate proposal by adding noise to parameters
        sigma = self.param_groups[0]['sigma']
        for group in self.param_groups:
            for p in group['params']:
                noise = torch.randn_like(p) * sigma
                p.add_(noise)

        # Compute proposed loss
        with torch.no_grad():
            proposed_loss = float(closure(False))

        delta_loss = proposed_loss - current_loss
        T = self.param_groups[0]['T']
        alpha = torch.exp(torch.tensor(-delta_loss / T)).item()
        accept = alpha >= 1.0 or torch.rand(1).item() < alpha

        # Accept or reject the proposal
        if accept:
            final_loss = proposed_loss
        else:
            # Restore parameters
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    p.copy_(saved_params[idx])
                    idx += 1
            final_loss = current_loss

        # Update acceptance rate with exponential moving average
        self.state['accept_rate'] = 0.9 * self.state['accept_rate'] + 0.1 * (1.0 if accept else 0.0)

        # Adapt sigma
        if self.param_groups[0]['adapt_sigma']:
            target_accept = self.param_groups[0]['target_accept']
            current_sigma = self.param_groups[0]['sigma']
            if self.state['accept_rate'] < target_accept:
                current_sigma *= 0.99
            else:
                current_sigma *= 1.01
            for group in self.param_groups:
                group['sigma'] = current_sigma

        # Decay temperature
        temperature_decay = self.param_groups[0]['temperature_decay']
        for group in self.param_groups:
            group['T'] *= temperature_decay

        self.state['step'] += 1

        return final_loss