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


class FFTPreconditioning2(Optimizer):
    """L"""
    def __init__(self, params, lr=1e-3, beta=0.9, epsilon=1e-8):
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                self._init_param_state(p)

    def _init_param_state(self, p):
        if p.requires_grad:
            shape = p.shape
            if len(shape) == 0:
                return  # Skip scalars

            # Handle 1D parameters as (1, n) instead of (n, 1)
            if len(shape) == 1:
                d0 = 1
                d_rest = shape[0]
            else:
                d0 = shape[0]
                d_rest = int(torch.prod(torch.tensor(shape[1:])).item())

            state = self.state[p]
            # FFT frequencies for rows (dim=1) and columns (dim=0)
            n_freq_rows = (d_rest // 2) + 1
            n_freq_cols = (d0 // 2) + 1

            state['left_power'] = torch.zeros(n_freq_rows, device=p.device, dtype=p.dtype)
            state['right_power'] = torch.zeros(n_freq_cols, device=p.device, dtype=p.dtype)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("FourierPreconditioner does not support sparse gradients")

                state = self.state[p]
                if 'left_power' not in state:
                    self._init_param_state(p)
                if len(state) == 0:
                    continue

                original_shape = grad.shape
                if grad.numel() == 1:
                    p.sub_(grad, alpha = lr)
                    continue  # Skip scalars

                # Reshape gradient to 2D (consistent with initialization)
                if len(original_shape) == 1:
                    grad_2d = grad.unsqueeze(0)  # (1, n)
                else:
                    grad_2d = grad.view(original_shape[0], -1)  # (d0, d_rest)

                # Update row-wise (dim=1) power spectrum
                fft_rows = torch.fft.rfft(grad_2d, dim=1)
                power_rows = (fft_rows.real.pow(2) + fft_rows.imag.pow(2)).mean(dim=0)
                if state['left_power'].shape != power_rows.shape:
                    # Reset if shape changes (unlikely in standard networks)
                    state['left_power'] = torch.zeros_like(power_rows)
                state['left_power'].mul_(beta).add_(power_rows, alpha=1 - beta)

                # Update column-wise (dim=0) power spectrum
                fft_cols = torch.fft.rfft(grad_2d, dim=0)
                power_cols = (fft_cols.real.pow(2) + fft_cols.imag.pow(2)).mean(dim=1)
                if state['right_power'].shape != power_cols.shape:
                    state['right_power'] = torch.zeros_like(power_cols)
                state['right_power'].mul_(beta).add_(power_cols, alpha=1 - beta)

                # Precondition rows (dim=1)
                scale_left = 1.0 / torch.sqrt(state['left_power'] + epsilon)
                scaled_fft_rows = fft_rows * scale_left.unsqueeze(0)
                left_preconditioned = torch.fft.irfft(scaled_fft_rows, dim=1, n=grad_2d.size(1))

                # Precondition columns (dim=0)
                fft_cols_precond = torch.fft.rfft(left_preconditioned, dim=0)
                scale_right = 1.0 / torch.sqrt(state['right_power'] + epsilon)
                scaled_fft_cols = fft_cols_precond * scale_right.unsqueeze(1)
                right_preconditioned = torch.fft.irfft(scaled_fft_cols, dim=0, n=grad_2d.size(0))

                # Reshape back to original dimensions
                if len(original_shape) == 1:
                    preconditioned_grad = right_preconditioned.squeeze(0)
                else:
                    preconditioned_grad = right_preconditioned.view(*original_shape)

                # Update parameters
                p.sub_(preconditioned_grad, alpha = lr)

        return loss