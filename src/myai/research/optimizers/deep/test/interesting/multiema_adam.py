import torch
from torch.optim import Optimizer

class MultiEMAAdam(Optimizer):
    """Implements a variant of Adam with multiple squared EMAs and higher-order differences for preconditioning.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (tuple of 4 floats, optional): coefficients for EMAs and their differences:
            - beta1: fast EMA decay rate for squared gradients
            - beta2: slow EMA decay rate for squared gradients
            - beta3: fast EMA decay rate for first difference
            - beta4: slow EMA decay rate for first difference
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        degree (int, optional): degree of the preconditioning formula (4 for quartic, 5 for quintic) (default: 4)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9, 0.999), eps=1e-8, weight_decay=0, degree=4):
        if degree not in (4, 5):
            raise ValueError("Degree must be 4 (quartic) or 5 (quintic)")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, degree=degree)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('MultiEMAAdam does not support sparse gradients')

                state = self.state[p]

                # Initialize state if necessary
                if len(state) == 0:
                    state['step'] = 0
                    # First-level EMAs for squared gradients
                    state['ema1_fast'] = torch.zeros_like(p.data)
                    state['ema1_slow'] = torch.zeros_like(p.data)
                    # Second-level EMAs for first difference
                    state['ema2_fast'] = torch.zeros_like(p.data)
                    state['ema2_slow'] = torch.zeros_like(p.data)

                # Retrieve parameters and state
                ema1_fast, ema1_slow = state['ema1_fast'], state['ema1_slow']
                ema2_fast, ema2_slow = state['ema2_fast'], state['ema2_slow']
                beta1, beta2, beta3, beta4 = group['betas']
                step = state['step']
                lr = group['lr']
                eps = group['eps']
                degree = group['degree']
                weight_decay = group['weight_decay']

                state['step'] += 1

                # Apply weight decay
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                # Update first-level EMAs (squared gradients)
                ema1_fast.mul_(beta1).addcmul_(grad, grad, value=1 - beta1)
                ema1_slow.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction for first-level EMAs
                bc_ema1_fast = 1 - beta1 ** state['step']
                bc_ema1_slow = 1 - beta2 ** state['step']
                ema1_fast_corr = ema1_fast / bc_ema1_fast
                ema1_slow_corr = ema1_slow / bc_ema1_slow

                # Compute first difference (absolute value to ensure positivity)
                delta1 = (ema1_slow_corr - ema1_fast_corr).abs()

                # Update second-level EMAs (first difference)
                ema2_fast.mul_(beta3).add_(delta1, alpha=1 - beta3)
                ema2_slow.mul_(beta4).add_(delta1, alpha=1 - beta4)

                # Bias correction for second-level EMAs
                bc_ema2_fast = 1 - beta3 ** state['step']
                bc_ema2_slow = 1 - beta4 ** state['step']
                ema2_fast_corr = ema2_fast / bc_ema2_fast
                ema2_slow_corr = ema2_slow / bc_ema2_slow

                # Compute second difference (absolute value)
                delta2 = (ema2_slow_corr - ema2_fast_corr).abs()

                # Compute denominator based on degree
                if degree == 4:
                    denom = (ema1_slow_corr * delta1).add_(eps).pow_(1/4)
                elif degree == 5:
                    denom = (ema1_slow_corr * delta1 * delta2).add_(eps).pow_(1/5)
                else:
                    raise NotImplementedError(f"Degree must be 4 (quartic) or 5 (quintic), got {degree}")

                # Update parameters
                if state['step'] < 20:
                    lr = lr * (1 / (20 - state['step']) ** 4)
                p.data.addcdiv_(grad, denom, value=-lr)

        return loss