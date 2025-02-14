import torch
from torch.optim import Optimizer

class ALOpt(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, epsilon=1e-8):
        defaults = dict(lr=lr, beta=beta, epsilon=epsilon)
        super().__init__(params, defaults)

        self.global_s_pos = 0.0
        self.global_s_neg = 0.0
        self.epsilon = epsilon

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        total_pos = 0.0
        total_neg = 0.0
        total_elements = 0

        # First pass: compute global positive/negative stats
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad_sign = torch.where(grad > 0, 1.0, -1.0)

                pos_count = (grad_sign == 1.0).sum().item()
                neg_count = (grad_sign == -1.0).sum().item()

                total_pos += pos_count
                total_neg += neg_count
                total_elements += grad.numel()

        if total_elements == 0:
            return loss

        # Update global statistics with moving average
        beta = self.defaults['beta']
        current_global_pos = total_pos / total_elements
        current_global_neg = total_neg / total_elements

        self.global_s_pos = beta * self.global_s_pos + (1 - beta) * current_global_pos
        self.global_s_neg = beta * self.global_s_neg + (1 - beta) * current_global_neg

        # Second pass: update parameters
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            epsilon = group.get('epsilon', self.epsilon)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if 's_pos' not in state:
                    state['s_pos'] = torch.zeros_like(p.data)
                if 's_neg' not in state:
                    state['s_neg'] = torch.zeros_like(p.data)

                s_pos = state['s_pos']
                s_neg = state['s_neg']

                # Compute gradient signs
                grad_sign = torch.where(grad > 0, 1.0, -1.0)
                mask_pos = (grad_sign == 1.0).float()
                mask_neg = (grad_sign == -1.0).float()

                # Update per-parameter support estimates
                s_pos.mul_(beta).add_((1 - beta) * mask_pos)
                s_neg.mul_(beta).add_((1 - beta) * mask_neg)

                # Compute support and global support
                support = torch.where(grad_sign == 1.0, s_pos, s_neg)
                global_support = torch.where(
                    grad_sign == 1.0,
                    torch.tensor(self.global_s_pos, device=grad.device),
                    torch.tensor(self.global_s_neg, device=grad.device)
                )

                # Calculate lift with epsilon smoothing
                lift = support / (global_support + epsilon)

                # Apply update
                p.data.add_(-lr * grad_sign * lift)

        return loss