import torch
from torch.optim import Optimizer

class LieSGD(Optimizer):
    """
    Implements the LieSGD, which utilizes the cross product from the
    Lie algebra so(3) to update parameters with rotational momentum.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate (default: 1e-3).
        alpha (float, optional): Momentum factor (default: 0.9).
        beta (float, optional): Scaling factor for the cross product term (default: 0.1).
    """

    def __init__(self, params, lr=1e-3, alpha=0.9, beta=0.1):
        defaults = dict(lr=lr, alpha=alpha, beta=beta)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LieAlgebraOptimizer does not support sparse gradients')

                # Flatten and pad gradient to multiples of 3
                grad_flat = grad.view(-1)
                num_elements = grad_flat.numel()
                pad_size = (3 - (num_elements % 3)) % 3
                grad_padded = torch.cat([grad_flat, torch.zeros(pad_size, device=grad.device)])
                grad_grouped = grad_padded.view(-1, 3)  # Shape (N, 3)

                # Retrieve state
                state = self.state[p]
                if 'velocity' not in state:
                    velocity = torch.zeros_like(grad_grouped)
                    state['velocity'] = velocity

                velocity = state['velocity']

                # Compute cross product (Lie bracket in so(3))
                cross = torch.cross(velocity, grad_grouped, dim=1)

                # Update velocity: v_{t+1} = αv_t + lr*(g_t + β*(v_t × g_t))
                updated_velocity = alpha * velocity + lr * (grad_grouped + beta * cross)

                # Update state with new velocity
                state['velocity'] = updated_velocity.detach()

                # Unpad and reshape to original gradient shape
                update_flat = updated_velocity.view(-1)[:num_elements]
                update = update_flat.view(grad.shape)

                # Apply parameter update: θ_{t+1} = θ_t - v_{t+1}
                p.data.sub_(update)

        return loss