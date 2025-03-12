import torch
from torch.optim import Optimizer

class TVO(Optimizer):
    """
    Implements the Two-Velocity Optimizer (TVO).

    It combines momentum with a secondary velocity that adapts to the curvature of the loss landscape by considering the change in gradients.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        betas (Tuple[float, float], optional): coefficients used for computing
            velocities (default: (0.9, 0.9))
        alpha (float, optional): coefficient for the secondary velocity term (default: 0.1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr, betas=(0.9, 0.9), alpha=0.1, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, alpha=alpha, weight_decay=weight_decay)
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
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TVO does not support sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Primary velocity
                    state['v1'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Secondary velocity
                    state['v2'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous gradient
                    state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                beta1, beta2 = group['betas']
                alpha = group['alpha']

                state['step'] += 1
                weight_decay = group['weight_decay']
                lr = group['lr']

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Primary velocity update
                v1 = state['v1']
                v1.mul_(beta1).add_(grad, alpha=-lr)

                # Secondary velocity update
                v2 = state['v2']
                prev_grad = state['prev_grad']
                v2.mul_(beta2).add_(prev_grad.sub(grad), alpha=lr)  # v2_t = beta2 * v2_{t-1} + lr * (g_{t-1} - g_t)

                # Parameter update
                p.add_(v1)
                p.add_(v2, alpha=alpha) # p_t = p_{t-1} + v1_t + alpha * v2_t

                # Store current gradient for next step (as previous gradient)
                state['prev_grad'].copy_(grad)


        return loss

