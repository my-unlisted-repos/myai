import torch
from torch.optim import Optimizer

class StochR(Optimizer):
    """
    rprop on EMA. max_step set to low value to make it more stable but might need increase to 1 in some cases.
    """
    def __init__(self, params, lr=1e-3, beta=0.9, increase_factor=1.2,
                 decrease_factor=0.5, max_step=0.1, min_step=1e-6):
        defaults = dict(lr=lr, beta=beta, increase_factor=increase_factor,
                        decrease_factor=decrease_factor, max_step=max_step,
                        min_step=min_step)
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step_size'] = torch.full_like(p.data, group['lr'])
                state['moving_avg'] = torch.zeros_like(p.data)
                state['previous_sign'] = torch.sign(torch.ones_like(p.data))  # Initialize to positive

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            beta = group['beta']
            increase = group['increase_factor']
            decrease = group['decrease_factor']
            max_step = group['max_step']
            min_step = group['min_step']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                m_avg = state['moving_avg']
                step_size = state['step_size']
                prev_sign = state['previous_sign']

                # Update moving average of gradients
                m_avg.mul_(beta).add_((1 - beta) * grad)

                # Get current sign of moving average
                current_sign = torch.sign(m_avg)

                # Determine sign changes (element-wise)
                sign_changed = current_sign != prev_sign

                # Adapt step sizes
                step_size.mul_(torch.where(sign_changed, decrease, increase))
                step_size.clamp_(min_step, max_step)

                # Update parameters: step = step_size * -sign(gradient)
                p.data.add_(-step_size * current_sign)

                # Store current sign for next iteration
                prev_sign.copy_(current_sign)

        return loss