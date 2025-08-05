import torch
from torch.optim import Optimizer
from collections import deque

def clamp(x, min, max):
    if x<min:x=min
    if x>max:x=max
    return x

class BundleGradientDescent(Optimizer):
    def __init__(self, params, lr=1e-3, m=2):
        if m < 2:
            raise ValueError("m must be at least 2")
        defaults = dict(lr=lr, m=m)
        super(BundleGradientDescent, self).__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['grad_buffer'] = deque(maxlen=m)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            m = group['m']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("BundleGradientDescent does not support sparse gradients")

                state = self.state[p]
                g_buf = state['grad_buffer']
                g_buf.append(grad.clone())

                if len(g_buf) < m:
                    direction = grad
                else:
                    g_prev = g_buf[0]
                    g_curr = g_buf[-1]

                    # Compute optimal combination
                    dot_product = torch.sum(g_prev * g_curr)
                    norm_prev_sq = torch.sum(g_prev ** 2)
                    norm_curr_sq = torch.sum(g_curr ** 2)

                    numerator = norm_curr_sq - dot_product
                    denominator = norm_prev_sq + norm_curr_sq - 2 * dot_product

                    if denominator < 1e-8:
                        alpha = 0.5
                    else:
                        alpha = numerator / denominator
                    alpha = clamp(alpha, 0.0, 1.0)

                    direction = alpha * g_prev + (1 - alpha) * g_curr

                p.data.add_(-lr * direction)

        return loss