# pylint:disable=not-callable
import torch
from torch.optim import Optimizer

class FFTProp(Optimizer):
    """like rmsprop but stores squared fft magnitude"""
    def __init__(self, params, lr=1e-3, beta=0.999, eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Compute FFT of the gradient
                grad_fft = torch.fft.rfftn(grad, dim=list(range(grad.dim())))

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Initialize exponential average for squared magnitudes
                    state['v'] = torch.zeros_like(grad_fft.real)

                state['step'] += 1
                v = state['v']
                beta = group['beta']
                eps = group['eps']

                # Compute squared magnitude of FFT coefficients
                squared_mags = grad_fft.real.pow(2) + grad_fft.imag.pow(2)

                # Update exponential moving average
                v.mul_(beta).add_(squared_mags, alpha=1 - beta)

                # Compute scaling factor
                denom = v.sqrt().add_(eps)

                # Scale the FFT coefficients by lr / denom
                scaling = group['lr'] / denom
                scaled_real = grad_fft.real * scaling
                scaled_imag = grad_fft.imag * scaling
                grad_fft_scaled = torch.complex(scaled_real, scaled_imag)

                # Compute inverse FFT to get the update direction
                update = torch.fft.irfftn(grad_fft_scaled, s=grad.shape, dim=list(range(grad.dim())))

                # Update parameters
                p.data.sub_(update)

        return loss