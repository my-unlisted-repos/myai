# pylint:disable=signature-differs, not-callable

import torch
from torch.optim import Optimizer

class StructuredQRSGD(Optimizer):
    """Transforms the gradient using the orthogonal matrix Q to align it with the column space of the parameter matrix"""
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('StructuredQROptimizer does not support sparse gradients')

                # Process based on parameter dimensionality
                if p.dim() == 1:
                    # Treat vector as a column matrix (n, 1)
                    param_matrix = p.data.view(-1, 1)
                    Q, R = torch.linalg.qr(param_matrix)
                    grad_matrix = grad.view(-1, 1)
                    grad_R = Q.T.mm(grad_matrix)
                    R_updated = R - lr * grad_R
                    param_updated = Q.mm(R_updated)
                    p.data = param_updated.view(-1)
                elif p.dim() == 2:
                    # Matrix parameter
                    Q, R = torch.linalg.qr(p.data)
                    grad_R = Q.T.mm(grad)
                    R_updated = R - lr * grad_R
                    p.data = Q.mm(R_updated)
                else:
                    # For higher dimensions, flatten to 2D and then reshape back
                    original_shape = p.data.shape
                    param_2d = p.data.view(original_shape[0], -1)
                    Q, R = torch.linalg.qr(param_2d)
                    grad_2d = grad.view(original_shape[0], -1)
                    grad_R = Q.T.mm(grad_2d)
                    R_updated = R - lr * grad_R
                    param_updated = Q.mm(R_updated)
                    p.data = param_updated.view(original_shape)
        return loss