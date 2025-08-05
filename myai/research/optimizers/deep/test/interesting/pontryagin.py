import torch
from torch.optim import Optimizer

class Pontryagin(Optimizer):
    """
    Implements the Pontryagin Optimizer, which adapts gradient updates using frequency-domain normalization.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        beta (float, optional): Exponential decay rate for gradient magnitude estimation (default: 0.9).
        eps (float, optional): Term added to denominator to improve numerical stability (default: 1e-8).
    """

    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("PontryaginOptimizer does not support sparse gradients")

                # Get FFT dimensions (all dimensions of the gradient tensor)
                dims = list(range(grad.dim()))

                # Compute real FFT of the gradient
                grad_fft = torch.fft.rfftn(grad, dim=dims)

                # Compute squared magnitude of FFT coefficients
                mag_sq = grad_fft.real.pow(2) + grad_fft.imag.pow(2)

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['S'] = torch.zeros_like(mag_sq)
                    state['beta'] = group['beta']

                # Update the exponential moving average of squared magnitudes
                state['S'].mul_(state['beta']).add_(mag_sq, alpha=1 - state['beta'])

                # Normalize the FFT coefficients
                denom = state['S'].add(group['eps']).sqrt()
                grad_fft_normalized = torch.complex(
                    grad_fft.real / denom,
                    grad_fft.imag / denom
                )

                # Compute inverse FFT to get the update direction
                update = torch.fft.irfftn(grad_fft_normalized, s=grad.shape, dim=dims)

                # Update parameters
                p.data.add_(update, alpha=-group['lr'])

        return loss