# pylint:disable=signature-differs, not-callable

import math

import torch
from torch.optim import Optimizer


class MorseFlow(Optimizer):
    """it was supposed to be more stable, but seems just worse in every way compared to SGD or Adam"""
    def __init__(self, params, lr=1e-3, momentum=0.9,
                 topo_scale=0.1, stability_threshold=60):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            topo_scale=topo_scale,
            stability_threshold=math.radians(stability_threshold)
        )
        super().__init__(params, defaults)

        # Initialize topology tracking states as tensors
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)
                state['grad_history'] = torch.zeros_like(p.data)
                state['angle_buffer'] = torch.zeros_like(p.data)
                state['stability_count'] = torch.zeros_like(p.data)  # Fix initialization

    def _vector_angle(self, a, b):
        """Compute cosine similarity between tensors"""
        return torch.sum(a * b) / (torch.norm(a) * torch.norm(b) + 1e-8)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            topo_scale = group['topo_scale']
            stability_threshold = group['stability_threshold']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad
                buf = state['momentum_buffer']
                angle_buf = state['angle_buffer']
                grad_hist = state['grad_history']
                stability_count = state['stability_count']

                # Update gradient history
                grad_hist.mul_(0.95).add_(grad, alpha=0.05)

                # stability metrics
                current_angle = self._vector_angle(grad, grad_hist)
                angle_diff = torch.abs(current_angle - angle_buf)
                angle_buf.copy_(current_angle)

                # Update stability counter
                stability_count.copy_(
                    torch.where(
                        angle_diff < stability_threshold,
                        stability_count + 1,
                        torch.zeros_like(stability_count)
                    )
                )

                # Adaptive momentum modulation
                stability_factor = torch.sigmoid(stability_count / 10 - 3)
                adaptive_momentum = momentum * (1 - stability_factor) + 0.5 * stability_factor

                # flow adjustment
                flow_direction = grad_hist * torch.sign(current_angle)
                topo_adjustment = topo_scale * flow_direction * stability_factor

                # Update momentum buffer
                buf.mul_(adaptive_momentum).add_(grad + topo_adjustment, alpha=1)

                # Final parameter update
                p.add_(buf, alpha=-lr)

        return loss