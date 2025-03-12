import torch
from torch.optim import Optimizer

class MultiscaleDescent(Optimizer):
    """
    adaptively scales low/high frequency gradient components in wavelet decomposition.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        alpha (float): Low-frequency scaling factor (default: 1.0).
        beta (float): High-frequency damping factor (default: 0.1).
        decomposition_levels (int): Number of wavelet decomposition levels (default: 1).
    """

    def __init__(self, params, lr=1e-3, alpha=1.0, beta=0.1, decomposition_levels=1):
        if decomposition_levels < 1:
            raise ValueError("Decomposition levels must be ≥ 1")
        defaults = dict(lr=lr, alpha=alpha, beta=beta, decomposition_levels=decomposition_levels)
        super().__init__(params, defaults)

    def _wavelet_decomposition(self, tensor, levels):
        """Multi-level Haar wavelet decomposition with padding handling"""
        coeffs = []
        current = tensor.view(-1)

        for _ in range(levels):
            n = current.size(0)
            even = current[::2]
            odd_indices = torch.arange(1, n, 2, device=tensor.device)
            odd = current[odd_indices] if odd_indices.numel() > 0 else torch.empty(0, device=tensor.device)

            # Handle odd-length tensors
            if even.size(0) != odd.size(0):
                odd = torch.cat([odd, torch.zeros(1, device=tensor.device, dtype=tensor.dtype)])

            # Compute low/high frequency components
            low = (even + odd) / 2
            high = (even - odd) / 2
            coeffs.append(high)
            current = low

        coeffs.append(current)  # Final low-frequency component
        return coeffs

    def _wavelet_reconstruction(self, coeffs, original_shape):
        """Inverse wavelet transform from decomposition coefficients"""
        current = coeffs[-1]

        for high in reversed(coeffs[:-1]):
            n_high = high.size(0)
            n_current = current.size(0)
            even = current + high
            odd = current - high

            # Interleave even and odd components
            reconstructed = torch.zeros(2 * n_current, device=current.device, dtype=current.dtype)
            reconstructed[::2] = even
            reconstructed[1::2] = odd[:n_high]  # Handle potential length mismatch

            # Preserve any remaining elements from odd array
            if n_high < n_current:
                remaining = odd[n_high:]
                reconstructed = torch.cat([reconstructed, remaining])

            current = reconstructed

        return current[:original_shape.numel()].view(original_shape)

    @torch.no_grad
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            beta = group['beta']
            levels = group['decomposition_levels']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MultiscaleDescent does not support sparse gradients")

                # Store original shape and decompose
                original_shape = grad.shape
                coeffs = self._wavelet_decomposition(grad, levels)

                # Apply frequency-dependent scaling
                scaled_coeffs = []
                scaled_coeffs.append(coeffs[0] * alpha)  # First low-frequency component
                for c in coeffs[1:-1]:
                    scaled_coeffs.append(c * beta)
                scaled_coeffs.append(coeffs[-1] * alpha)  # Final low-frequency component

                # Reconstruct modified gradient
                modified_grad = self._wavelet_reconstruction(scaled_coeffs, original_shape)

                # Update parameters
                p.add_(-lr * modified_grad)

        return loss