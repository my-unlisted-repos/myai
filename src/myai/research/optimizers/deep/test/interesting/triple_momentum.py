import torch
from torch.optim import Optimizer

class TripleMomentum(Optimizer):
    """The Fastest Known Globally Convergent First-Order
Method for Minimizing Strongly Convex Functions
Bryan Van Scoy, Student Member, IEEE, Randy A. Freeman, Senior Member, IEEE,
and Kevin M. Lynch, Fellow, IEEE"""
    def __init__(self, params, lr=1e-3, mu=0.02):
        L = 1/lr
        self.L = L
        self.mu = mu
        kappa = mu / L
        rho = 1 - torch.sqrt(torch.tensor(kappa))
        alpha = (1 + rho) / L
        beta = rho**2 / (2 - rho)
        gamma = rho**2 / ((1 + rho) * (2 - rho))
        delta = rho**2 / (1 - rho**2)

        self.alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
        self.beta = beta.item() if isinstance(beta, torch.Tensor) else beta
        self.gamma = gamma.item() if isinstance(gamma, torch.Tensor) else gamma
        self.delta = delta.item() if isinstance(delta, torch.Tensor) else delta

        defaults = dict(L=L, mu=mu)
        super().__init__(params, defaults)

        # Initialize state variables
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['xsi_prev'] = torch.clone(p)
                state['xsi_current'] = torch.clone(p)
                state['y'] = torch.clone(p)

    @torch.no_grad
    def step(self, closure=None):
        # Temporarily set parameters to y
        params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        orig = [p.clone() for p in params]
        for p in params:
            p.copy_(self.state[p]['y'])

        with torch.enable_grad(): loss = closure()

        for p, param_copy in zip(params, orig):
            p.copy_(param_copy)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                xsi_prev = state['xsi_prev']
                xsi_current = state['xsi_current']
                y = state['y']

                # Compute xsi_next
                xsi_next = (1 + self.beta) * xsi_current - self.beta * xsi_prev - self.alpha * grad
                # Compute y_next
                y_next = (1 + self.gamma) * xsi_next - self.gamma * xsi_current
                # Compute x_next
                x_next = (1 + self.delta) * xsi_next - self.delta * xsi_current

                # Update parameter tensor
                p.data.copy_(x_next)

                # Update state for next iteration
                state['xsi_prev'] = xsi_current.clone()
                state['xsi_current'] = xsi_next.clone()
                state['y'] = y_next.clone()