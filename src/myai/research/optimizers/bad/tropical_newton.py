#type:ignore
import torch
import numpy as np
from torch.optim import Optimizer
from collections import deque,defaultdict

class TropicalNewton(Optimizer):
    """dododododododoododod"""
    def __init__(self, params, lr=0.01, buffer_size=2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if buffer_size < 2:
            raise ValueError("Buffer size must be at least 2.")

        defaults = dict(lr=lr, buffer_size=buffer_size)
        super().__init__(params, defaults)

        self.state = defaultdict(dict)
        self.state['buffer'] = deque(maxlen=buffer_size)

    def _gather_flat_params(self):
        params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p.data.view(-1))
        return torch.cat(params) if params else torch.tensor([])

    def _gather_flat_grad(self):
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads.append(p.grad.data.view(-1))
        return torch.cat(grads) if grads else torch.tensor([])

    def _distribute_flat_params(self, flat_params):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                numel = p.data.numel()
                p.data.copy_(flat_params[offset:offset+numel].view_as(p.data))
                offset += numel

    @torch.no_grad
    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required to compute loss.")

        with torch.enable_grad(): loss = closure()
        flat_params = self._gather_flat_params()
        flat_grad = self._gather_flat_grad()

        if flat_params.numel() == 0 or flat_grad.numel() == 0:
            return loss

        buffer = self.state['buffer']

        # Calculate parameter change from previous step and update stored c_i terms
        if hasattr(self, 'prev_flat_params'):
            delta_theta = flat_params - self.prev_flat_params
            for i in range(len(buffer)):
                v, c = buffer[i]
                delta_c = torch.dot(v, delta_theta).item()
                new_c = c + delta_c
                buffer[i] = (v, new_c)

        # Store current parameters for next delta calculation
        self.prev_flat_params = flat_params.clone()

        # Add new monomial with current gradient
        current_v = flat_grad.clone()
        current_c = loss.item() - torch.dot(current_v, flat_params).item()
        buffer.append((current_v, current_c))

        # Get monomial values at current parameters (should now be accurate)
        monomial_values = [torch.dot(v, flat_params).item() + c for v, c in buffer]

        if len(monomial_values) < 2:
            new_flat_params = flat_params - self.param_groups[0]['lr'] * flat_grad
            self._distribute_flat_params(new_flat_params)
            return loss

        # Find top two active monomials
        top_two = np.argsort(monomial_values)[-2:]
        v1, c1 = buffer[top_two[1]]
        v2, c2 = buffer[top_two[0]]

        # Compute Newton direction
        delta_dir = v1 - v2
        dir_norm_sq = torch.dot(delta_dir, delta_dir).item()

        if dir_norm_sq < 1e-10:
            new_flat_params = flat_params - self.param_groups[0]['lr'] * flat_grad
            self._distribute_flat_params(new_flat_params)
            return loss

        # Calculate optimal step size
        numerator = (c2 - c1) + torch.dot(flat_params, v2 - v1).item()
        alpha = numerator / dir_norm_sq

        # Apply update with learning rate
        step = alpha * delta_dir
        new_flat_params = flat_params - self.param_groups[0]['lr'] * step
        self._distribute_flat_params(new_flat_params)

        return loss