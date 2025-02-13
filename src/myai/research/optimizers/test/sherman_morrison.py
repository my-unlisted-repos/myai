import torch
from torch.optim import Optimizer

class ShermanMorrison(Optimizer):
    """
    Rank-One Preconditioned Optimizer (ROP).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): EMA decay factor for gradient moments (default: 0.9)
        alpha (float, optional): preconditioning strength (default: 0.1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Tuning:
        lr generally higher like 1e-1

    TODO:
        test alpha 0.3 - 0.5

    """

    def __init__(self, params, lr=1e-1, beta1=0.9, alpha=0.1, weight_decay=0):

        defaults = dict(lr=lr, beta1=beta1, alpha=alpha, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Initialize EMA vectors for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            alpha = group['alpha']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ROP does not support sparse gradients")

                state = self.state[p]
                v = state['v']

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update EMA vector v
                v.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Flatten tensors for vector operations
                v_flat = v.view(-1)
                grad_flat = grad.view(-1)

                # Compute v norm squared and dot product
                v_norm_sq = torch.dot(v_flat, v_flat)
                dot_product = torch.dot(v_flat, grad_flat)

                if v_norm_sq == 0 or alpha == 0:
                    # No preconditioning if v is zero or alpha is zero
                    p_grad = grad
                else:
                    # Sherman-Morrison update for rank-one preconditioning
                    scale = (alpha / (1 + alpha * v_norm_sq)) * dot_product
                    p_grad = grad - scale * v

                # Update parameters
                p.add_(p_grad, alpha=-lr)

        return loss