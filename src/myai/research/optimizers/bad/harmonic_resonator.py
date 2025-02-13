# pylint:disable=signature-differs, not-callable

import torch
from torch.optim import Optimizer

class HarmonicResonatorOptimizer(Optimizer):
    """well that sounded good but it just doesnt converge well"""
    def __init__(self, params, lr=1e-3, resonance_factor=0.9, damping=0.1, eps=1e-8):
        defaults = dict(lr=lr, resonance_factor=resonance_factor, damping=damping, eps=eps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['phase'] = torch.zeros_like(p)  # Wave-like memory
                state['velocity'] = torch.zeros_like(p)  # Oscillation velocity

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            resonance = group['resonance_factor']
            damping = group['damping']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                phase = state['phase']
                velocity = state['velocity']

                # Update velocity with resonance and damping
                velocity.mul_(resonance).add_(-damping * phase + grad)

                # Update phase (wave memory)
                phase.add_(velocity)

                # Calculate resonant update (amplify in-phase components)
                resonance_strength = torch.sum(phase * grad) / (torch.norm(phase)*torch.norm(grad) + eps)
                update = lr * (grad + resonance_strength * phase)

                # Apply update
                p.sub_(update)

        return loss