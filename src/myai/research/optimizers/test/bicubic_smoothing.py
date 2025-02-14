import torch
from torch.optim import Optimizer
import torch.nn.functional as F

class BiGS(Optimizer):
    """Bicubic Gradient Smoothing

    Applies bicubic interpolation to smooth gradients of multi-dimensional parameters,
    then performs a SGD update.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate. Default: 1e-3.
    """

    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("BicubicOptimizer does not support sparse gradients")

                # Smooth gradients for parameters with at least two dimensions
                if grad.dim() >= 2:
                    original_shape = grad.shape
                    # Reshape to (batch, channels, H, W) for interpolate
                    grad_reshaped = grad.view(-1, 1, *original_shape[-2:])
                    # Apply bicubic interpolation to smooth gradients
                    smoothed_grad = F.interpolate(
                        grad_reshaped,
                        size=original_shape[-2:],
                        mode='bicubic',
                        align_corners=False
                    )
                    # Reshape back to original dimensions
                    smoothed_grad = smoothed_grad.view(original_shape)
                    # Update the gradient with the smoothed version
                    p.add_(smoothed_grad, alpha=-lr)

                # Perform SGD update
                else:
                    p.add_(p.grad, alpha=-lr)

        return loss