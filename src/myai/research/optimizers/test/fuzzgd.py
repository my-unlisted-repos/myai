import math

import torch
from torch.optim import Optimizer


def triangular(x: float, peak: float, left_width: float, right_width: float) -> float:
    left_bound = peak - left_width
    right_bound = peak + right_width

    if x < left_bound or x > right_bound:
        return 0.0
    if x < peak:
        return (x - left_bound) / (peak - left_bound)
    return (right_bound - x) / (right_bound - peak)

class FuzzGD(Optimizer):
    """untested yet"""
    def __init__(self, params, lr=1e-3, ema_alpha=0.1, eps=1e-8):
        defaults = dict(lr=lr, ema_alpha=ema_alpha, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            ema_alpha = group['ema_alpha']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("FuzzyLogicOptimizer does not support sparse gradients")

                current_magnitude = grad.norm().item()
                state = self.state[p]

                if 'ema_mean' not in state:
                    # Initialize state
                    state['ema_mean'] = current_magnitude
                    state['ema_var'] = 0.0
                    state['prev_ema_mean'] = current_magnitude
                    adj = 1.0
                else:
                    # Retrieve previous values
                    prev_ema_mean = state['prev_ema_mean']
                    prev_ema_var = state['ema_var']

                    # Update EMA statistics
                    new_ema_mean = ema_alpha * current_magnitude + (1 - ema_alpha) * prev_ema_mean
                    delta = current_magnitude - prev_ema_mean
                    new_ema_var = ema_alpha * (delta ** 2) + (1 - ema_alpha) * prev_ema_var

                    # Calculate trend and z-scores
                    trend = current_magnitude - prev_ema_mean
                    ema_std = math.sqrt(prev_ema_var + eps)

                    z_mag = (current_magnitude - prev_ema_mean) / ema_std if ema_std != 0 else 0.0
                    z_trend = trend / ema_std if ema_std != 0 else 0.0

                    # Fuzzify inputs
                    mag_low = triangular(z_mag, -1.0, 2.0, 1.0)
                    mag_medium = triangular(z_mag, 0.0, 1.5, 1.5)
                    mag_high = triangular(z_mag, 1.0, 1.0, 2.0)

                    trend_dec = triangular(z_trend, -1.0, 2.0, 1.0)
                    trend_steady = triangular(z_trend, 0.0, 1.0, 1.0)
                    trend_inc = triangular(z_trend, 1.0, 1.0, 2.0)

                    # Define fuzzy rules (antecedents and consequents)
                    rules = [
                        (min(mag_low, trend_dec), 1.2),
                        (min(mag_low, trend_steady), 1.2),
                        (min(mag_low, trend_inc), 1.0),
                        (min(mag_medium, trend_dec), 1.2),
                        (min(mag_medium, trend_steady), 1.0),
                        (min(mag_medium, trend_inc), 0.8),
                        (min(mag_high, trend_dec), 1.0),
                        (min(mag_high, trend_steady), 0.8),
                        (min(mag_high, trend_inc), 0.8),
                    ]

                    # Calculate adjustment factor
                    total_strength = 0.0
                    weighted_sum = 0.0
                    for strength, consequent in rules:
                        total_strength += strength
                        weighted_sum += strength * consequent

                    adj = weighted_sum / total_strength if total_strength > 0 else 1.0

                    # Update state with new EMA values
                    state['ema_mean'] = new_ema_mean
                    state['ema_var'] = new_ema_var
                    state['prev_ema_mean'] = new_ema_mean

                # Update parameters with fuzzy-adjusted learning rate
                p.data.add_(grad, alpha=-lr * adj)

        return loss