import torch
from torch.optim import Optimizer

class CurvatureAwareMomentum(Optimizer):
    """
    Curvature-Aware Momentum (CAM) usually requires higher learnign rates like 1e-1 or 1.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): EMA decay rate for gradients (default: 0.9)
        beta2 (float, optional): EMA decay rate for curvature estimates (default: 0.999)
        epsilon (float, optional): term to improve numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['m'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                state['h'] = torch.ones_like(p.data, memory_format=torch.preserve_format)
                state['v_prev'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                m = state['m']
                h = state['h']
                v_prev = state['v_prev']
                step = state['step']

                # Update step counter
                state['step'] += 1

                # Compute delta_g for steps after the first
                if step > 0:
                    delta_g = grad - m  # Previous EMA (before update)

                # Update EMA of gradients
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update curvature estimate after the first step
                if step > 0:
                    delta_param = lr * v_prev
                    delta_param_abs = delta_param.abs().add_(epsilon)

                    # Compute curvature estimate and apply absolute value
                    curvature_estimate = delta_g.abs().div_(delta_param_abs)

                    # Update EMA of curvature estimate
                    h.mul_(beta2).add_(curvature_estimate, alpha=1 - beta2)

                # Compute denominator with stability epsilon
                denom = h.add(epsilon)

                # Compute current step direction
                v = m.div_(denom)

                # Update parameters
                p.add_(v, alpha=-lr)

                # Save current step direction for next iteration
                v_prev.copy_(v)

        return loss