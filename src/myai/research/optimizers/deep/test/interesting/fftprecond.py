# pylint:disable=not-callable
import torch
from torch.optim import Optimizer

class FFTPreconditioning(Optimizer):
    """
    Implements a novel FFT-based decomposition optimizer.

    The optimizer transforms gradients into the frequency domain, scales components
    inversely with their frequency magnitudes, and applies momentum in the spatial domain.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        momentum (float, optional): Momentum factor (default: 0.9).
        scaling_factor (float, optional): Frequency scaling strength (default: 1.0).
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, scaling_factor=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, scaling_factor=scaling_factor)
        super().__init__(params, defaults)

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
                    raise RuntimeError("FFTOptimizer does not support sparse gradients")

                state = self.state[p]

                # Initialize state if needed
                if 'frequency_mask' not in state:
                    state['frequency_mask'] = self._create_frequency_mask(grad.shape, grad.device)
                if 'momentum_buffer' not in state and group['momentum'] != 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                # FFT-based gradient processing
                processed_grad = self._process_gradient(grad, state, group['scaling_factor'])

                # Update parameters with momentum
                if group['momentum'] != 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(processed_grad)
                    p.data.add_(buf, alpha=-group['lr'])
                else:
                    p.data.add_(processed_grad, alpha=-group['lr'])

        return loss

    def _process_gradient(self, grad, state, scaling_factor):
        """Applies FFT decomposition and frequency-based scaling to gradients."""
        # Compute FFT and shift to center
        grad_fft = torch.fft.fftn(grad)
        grad_fft_shift = torch.fft.fftshift(grad_fft)

        # Apply frequency-dependent scaling
        frequency_mask = state['frequency_mask']
        scaled_fft = grad_fft_shift / (1.0 + scaling_factor * frequency_mask)

        # Shift back and inverse transform
        grad_fft_scaled = torch.fft.ifftshift(scaled_fft)
        processed_grad = torch.fft.ifftn(grad_fft_scaled).real

        return processed_grad

    def _create_frequency_mask(self, shape, device):
        """Creates n-dimensional frequency distance mask."""
        grids = []
        for i, dim_size in enumerate(shape):
            coords = torch.arange(dim_size, dtype=torch.float32, device=device)
            coords -= (dim_size // 2)  # Center coordinates at zero

            # Reshape for broadcasting
            view_shape = [1]*len(shape)
            view_shape[i] = -1
            coords = coords.view(view_shape)

            # Expand to match parameter dimensions
            grids.append(coords.expand(shape))

        # Calculate Euclidean distance from center
        squared_dist = torch.zeros(shape, device=device)
        for grid in grids:
            squared_dist += grid.pow(2)

        return torch.sqrt(squared_dist)