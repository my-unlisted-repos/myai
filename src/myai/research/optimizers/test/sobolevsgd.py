import torch
from torch.optim import Optimizer

class SobolevSGD(Optimizer):
    """
    Sobolev Gradient Descent. This should be more lightweight than FFT-SGD, and maybe less agressive. TODO use torch.gradient

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        lambda_ (float, optional): Smoothing strength (default: 0.1).
    """

    def __init__(self, params, lr=1e-3, lambda_=0.1):
        defaults = dict(lr=lr, lambda_=lambda_)
        super().__init__(params, defaults)

    @staticmethod
    def compute_laplacian(gradient):
        """Computes the discrete Laplacian of a gradient tensor across all dimensions."""
        laplacian = torch.zeros_like(gradient)
        if gradient.ndim == 0:  # Skip scalar tensors
            return laplacian
        for dim in range(gradient.ndim):
            # Define slices to extract left, center, and right elements along the current dimension
            slices_left = [slice(None)] * gradient.ndim
            slices_left[dim] = slice(0, -2)
            left = gradient[slices_left]

            slices_center = [slice(None)] * gradient.ndim
            slices_center[dim] = slice(1, -1)
            center = gradient[slices_center]

            slices_right = [slice(None)] * gradient.ndim
            slices_right[dim] = slice(2, None)
            right = gradient[slices_right]

            # Compute second difference and pad to original shape
            diff = left - 2 * center + right
            pad_size = list(gradient.shape)
            pad_size[dim] = 1
            pad = torch.zeros(pad_size, dtype=gradient.dtype, device=gradient.device)
            diff_padded = torch.cat([pad, diff, pad], dim=dim)
            laplacian += diff_padded
        return laplacian

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            lambda_ = group['lambda_']
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                laplacian = self.compute_laplacian(grad)
                # Combine gradient with Laplacian smoothing
                smoothed_grad = grad + lambda_ * laplacian
                # Update parameters
                param.data.add_(-lr * smoothed_grad)

        return loss