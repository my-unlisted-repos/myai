from collections.abc import Callable

import torch
import torch.optim as optim
from torch import Tensor

# bro had a vision
Params = list[Tensor]
MaybeGrad = Tensor | None
LossClosure = Callable | None
State = dict

class SkewAdam(optim.Optimizer):
    r"""Weird ass adam.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        skew_gamma (float, optional): strength of the skew-symmetric interaction
            term. Positive values introduce neighbor interactions. (default: 0.0)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) Not implemented for the skew term yet.

    The update rule for parameter `p` with gradient `g`, momentum `m`, variance `v` is:

    1. g_t = grad
    2. m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    3. v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    4. m_hat_t = m_t / (1 - beta1^t)
    5. v_hat_t = v_t / (1 - beta2^t)

    6. adam_term = m_hat_t / (sqrt(v_hat_t) + eps)

    7. # Skew term calculation (applied per tensor)
    8. m_flat = m_hat_t.flatten()
    9. m_rolled_plus = torch.roll(m_flat, shifts=-1, dims=0)
    10. m_rolled_minus = torch.roll(m_flat, shifts=1, dims=0)
    11. # Handle boundaries (set roll-over elements to zero)
    12. m_rolled_plus[-1] = 0.0
    13. m_rolled_minus[0] = 0.0
    14. skew_update_flat = skew_gamma * (m_rolled_plus - m_rolled_minus)
    15. skew_term = skew_update_flat.view_as(p)

    16. update = adam_term + skew_term
    17. p_new = p - lr * update
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, skew_gamma=0.0, amsgrad=False):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, skew_gamma=skew_gamma, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        # Set amsgrad default to False unless explicitly set in state
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad
    def step(self, closure: LossClosure = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            skew_gamma = group['skew_gamma']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('SkewAdam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])


            # Perform main Adam update
            self._adam_update(params_with_grad,
                              grads,
                              exp_avgs,
                              exp_avg_sqs,
                              max_exp_avg_sqs, # Note: AMSGrad part only applies to denom
                              state_steps,
                              group['amsgrad'],
                              beta1,
                              beta2,
                              group['lr'],
                              group['weight_decay'],
                              group['eps'],
                              skew_gamma)

        return loss

    def _adam_update(self, params: Params,
                     grads: Params,
                     exp_avgs: Params,
                     exp_avg_sqs: Params,
                     max_exp_avg_sqs: Params,
                     state_steps: list[int],
                     amsgrad: bool,
                     beta1: float,
                     beta2: float,
                     lr: float,
                     weight_decay: float,
                     eps: float,
                     skew_gamma: float):
        """ Functional API for Adam algorithm """

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            # Perform stepweight decay
            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            m_hat = exp_avg / bias_correction1
            v_hat = exp_avg_sq / bias_correction2

            # Denominator term (standard Adam)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                max_exp_avg_sq = max_exp_avg_sqs[i]
                torch.maximum(max_exp_avg_sq, v_hat, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sq.sqrt().add_(eps)
            else:
                denom = v_hat.sqrt().add_(eps)

            # Standard Adam update part
            adam_term = m_hat / denom

            # --- Skew Interaction Term ---
            skew_term = torch.zeros_like(param) # Initialize as zero
            if skew_gamma != 0.0 and param.numel() > 1:
                 # Flatten momentum estimate
                m_hat_flat = m_hat.flatten()
                num_elements = m_hat_flat.numel()

                if num_elements > 1:
                    # Roll to get m[i+1] and m[i-1]
                    m_rolled_plus = torch.roll(m_hat_flat, shifts=-1, dims=0)
                    m_rolled_minus = torch.roll(m_hat_flat, shifts=1, dims=0)

                    # Handle boundaries: difference at boundary uses only the neighbor inside
                    # Effectively, m[-1] = 0 and m[N] = 0
                    m_rolled_plus[-1] = 0.0
                    m_rolled_minus[0] = 0.0

                    # Calculate skew update: gamma * (m[i+1] - m[i-1])
                    skew_update_flat = skew_gamma * (m_rolled_plus - m_rolled_minus)

                    # Reshape back to original parameter shape
                    skew_term = skew_update_flat.view_as(param)

            # --- Combine terms ---
            update_direction = adam_term + skew_term

            # Apply final update
            param.add_(update_direction, alpha=-lr)

