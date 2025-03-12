import torch
from torch.optim import Optimizer

class HMM(Optimizer):
    """
    hmmmmmm...

    SGD like but beats sgd on rosen and since it doesnt have momentum thats interesting.

    Args:
        params (iterable): Parameters to optimize
        lr (float): Base learning rate (default: 1e-3)
        trust_decay (float): Trust state persistence (default: 0.95)
        distrust_decay (float): Distrust state persistence (default: 0.8)
        min_scale (float): Minimum learning rate multiplier (default: 0.1)
        max_scale (float): Maximum learning rate multiplier (default: 2.0)
        transition_temp (float): Transition probability sharpness (default: 100.0)
    """

    def __init__(self, params, lr=1e-3, trust_decay=0.95, distrust_decay=0.8,
                 min_scale=0.1, max_scale=2.0, transition_temp=100.0):
        if not 0.0 < trust_decay < 1.0 or not 0.0 < distrust_decay < 1.0:
            raise ValueError("Decay parameters must be in (0,1)")

        defaults = dict(lr=lr, trust_decay=trust_decay,
                       distrust_decay=distrust_decay, min_scale=min_scale,
                       max_scale=max_scale, transition_temp=transition_temp)
        super().__init__(params, defaults)

        # Initialize HMM states for each parameter element
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['prev_grad'] = torch.zeros_like(p, requires_grad=False)
                state['belief'] = torch.full_like(p, 0.5, requires_grad=False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            t_decay = group['trust_decay']
            d_decay = group['distrust_decay']
            min_lr = group['min_scale'] * lr
            max_lr = group['max_scale'] * lr
            temp = group['transition_temp']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                prev_grad = state['prev_grad']
                belief = state['belief']

                # Element-wise transition probabilities
                grad_product = grad * prev_grad
                similarity = torch.sigmoid(temp * grad_product)

                # State transition matrix components
                p_remain_trust = t_decay + (1 - t_decay) * similarity
                p_distrust_to_trust = (1 - d_decay) * similarity

                # HMM belief update
                new_belief = belief * p_remain_trust + (1 - belief) * p_distrust_to_trust
                new_belief.clamp_(0.0, 1.0)

                # Element-wise learning rates
                lr_scales = min_lr + (max_lr - min_lr) * new_belief

                # Update parameters with element-wise learning rates
                p.data -= lr_scales * grad

                # Maintain state
                state['prev_grad'].copy_(grad)
                state['belief'] = new_belief

        return loss