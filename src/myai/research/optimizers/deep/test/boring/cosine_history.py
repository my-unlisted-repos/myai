from collections import deque

import torch
from torch.optim import Optimizer


class CosineHistory(Optimizer):
    """weighted momentum with historical updates by cosine similarities need to test different history sizes"""
    def __init__(self, params, lr=1e-3, alpha=0.5, history_size=3, temperature=1.0):
        defaults = dict(lr=lr, alpha=alpha, history_size=history_size, temperature=temperature)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['history'] = deque(maxlen=group['history_size'])

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            temperature = group['temperature']
            history_size = group['history_size']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Initialize history if not present
                if 'history' not in state:
                    state['history'] = deque(maxlen=history_size)

                # Compute current gradient and flatten
                current_grad = grad.detach().clone()
                flat_current_grad = current_grad.flatten()

                # Compute similarities with historical gradients
                similarities = []
                past_updates = []
                for hist_grad, hist_update in state['history']:
                    flat_hist_grad = hist_grad.flatten()
                    dot_product = torch.dot(flat_current_grad, flat_hist_grad)
                    norm_product = torch.norm(flat_current_grad) * torch.norm(flat_hist_grad)
                    similarity = dot_product / (norm_product + 1e-8)
                    similarities.append(similarity)
                    past_updates.append(hist_update)

                # Compute weights using softmax with temperature
                if similarities:
                    similarities = torch.stack(similarities)
                    weights = torch.softmax(similarities / temperature, dim=0)
                    # Compute weighted sum of past updates
                    weighted_past_updates = sum(w * u for w, u in zip(weights, past_updates))
                else:
                    weighted_past_updates = torch.zeros_like(p.data)

                # Compute current update
                current_update = lr * current_grad + alpha * weighted_past_updates

                # Apply the update
                p.data.add_(-current_update)

                # Store current gradient and update in history
                state['history'].append( (current_grad.clone(), current_update.clone()) )

        return loss