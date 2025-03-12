import torch
from torch.optim import Optimizer

class StochasticSteffensen(Optimizer):
    """
    Implements the Stochastic Steffensen optimizer.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        alpha (float, optional): perturbation scaling factor (default: 1e-3)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, alpha=1e-3, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(StochasticSteffensen, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            raise ValueError("Closure required for StochasticSteffensen")

        # First pass: compute loss and gradients at current parameters (g0)
        with torch.enable_grad(): loss = closure()

        # Save original parameters and gradients
        original_params = []
        original_grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                original_params.append(p.clone(memory_format=torch.preserve_format))
                original_grads.append(p.grad.clone())

        # Perturb parameters: p += alpha * g0
        index = 0
        for group in self.param_groups:
            alpha = group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                g0 = original_grads[index]
                p.add_(alpha * g0)
                index += 1

        # Second pass: compute gradients at perturbed parameters (g1)
        # Zero gradients before backward pass
        self.zero_grad()
        with torch.enable_grad(): perturbed_loss = closure()
        # perturbed_loss.backward()

        # Restore original parameters and compute updates
        index = 0
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g0 = original_grads[index]
                g1 = p.grad.clone()
                p.copy_(original_params[index])

                # Compute denominator with epsilon for numerical stability
                denominator = g1 - g0
                denominator.add_(torch.sign(denominator) * eps)

                # Element-wise update calculation
                update = (group['alpha'] * g0**2) / denominator

                # Apply update
                p.sub_(lr * update)
                index += 1

        return perturbed_loss