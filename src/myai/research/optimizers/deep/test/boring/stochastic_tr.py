import torch
from torch.optim import Optimizer

class StochasticTrustRegion(Optimizer):
    """Stochastic Trust Region."""
    def __init__(self, params, delta_init=1.0, eta1=0.1, eta2=0.7, gamma_inc=2.0, gamma_dec=0.5,
                 delta_min=1e-6, delta_max=1e6):
        """
        Args:
            params (iterable): Model parameters.
            delta_init (float): Initial trust region radius.
            eta1 (float): Threshold for accepting a step.
            eta2 (float): Threshold for increasing delta.
            gamma_inc (float): Factor to increase delta.
            gamma_dec (float): Factor to decrease delta.
            delta_min (float): Minimum delta value.
            delta_max (float): Maximum delta value.
        """
        defaults = dict(delta=delta_init)
        super().__init__(params, defaults)
        self.eta1 = eta1
        self.eta2 = eta2
        self.gamma_inc = gamma_inc
        self.gamma_dec = gamma_dec
        self.delta_min = delta_min
        self.delta_max = delta_max

    @torch.no_grad
    def step(self, closure):
        """
        Performs one optimization step.

        Args:
            closure (callable): Computes loss and gradients, returning the loss.
        """
        # Compute loss and gradients at current point
        with torch.enable_grad(): loss_k = closure()  # This also populates param.grad

        # Flatten gradients into a vector
        params = self.param_groups[0]['params']
        g_k = torch.cat([p.grad.view(-1) for p in params if p.grad is not None])
        norm_g_k = torch.norm(g_k)

        if norm_g_k == 0:
            return loss_k  # Skip if gradient is zero

        delta_k = self.state.get('delta', self.param_groups[0]['delta'])

        # Compute step
        s_k = -(delta_k / norm_g_k) * g_k

        # Store original parameters for potential revert
        originals = [p.data.clone() for p in params]

        # Apply step
        offset = 0
        for p in params:
            if p.grad is None:
                continue
            numel = p.numel()
            p.data.add_(s_k[offset:offset + numel].view_as(p))
            offset += numel

        # Compute loss at tentative new point
        loss_kp1 = closure(False)

        # Compute improvement ratio
        pred = delta_k * norm_g_k
        ared = loss_k - loss_kp1
        rho = ared / pred if pred != 0 else float('-inf')

        # Update trust region and accept/reject step
        if rho > self.eta1:
            # Accept step
            if rho > self.eta2:
                delta_k *= self.gamma_inc
            # Else delta_k remains unchanged
        else:
            # Reject step, revert parameters
            for p, orig in zip(params, originals):
                if p.grad is not None:
                    p.data.copy_(orig)
            delta_k *= self.gamma_dec

        # Clip delta
        delta_k = max(self.delta_min, min(self.delta_max, delta_k))
        self.state['delta'] = delta_k

        return loss_k