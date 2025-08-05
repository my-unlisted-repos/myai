import torch
from torch.optim import Optimizer

class CubicAdam(Optimizer):
    """based on diagmin"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
                state['param_history'] = []
                state['grad_history'] = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                state['step'] += 1

                # Update momentum and variance estimates
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute Adam-style scaled gradient
                denom = exp_avg_sq.sqrt().add_(eps)
                scaled_grad = exp_avg / denom

                # Store current parameter and gradient
                state['param_history'].append(p.data.clone())
                state['grad_history'].append(scaled_grad.clone())

                # Maintain history of last four steps
                if len(state['param_history']) > 4:
                    state['param_history'].pop(0)
                    state['grad_history'].pop(0)

                # Perform cubic fitting every four steps
                if len(state['param_history']) == 4:
                    params = torch.stack(state['param_history'])
                    w0, w1, w2, w3 = params

                    # Calculate cubic coefficients
                    w1_diff = w1 - w0
                    w2_diff = w2 - w0
                    w3_diff = w3 - w0

                    a =  1.5*w1_diff - 0.6*w2_diff + 0.1*w3_diff
                    b = -2.5*w1_diff + 1.2*w2_diff - 0.1*w3_diff
                    c =  1.0*w1_diff - 0.6*w2_diff + 0.1*w3_diff

                    # Find minima of cubic model
                    discriminant = (2*b)**2 - 12*a*c
                    sqrt_disc = torch.sqrt(torch.abs(discriminant))
                    t1 = (-2*b + sqrt_disc) / (6*a + 1e-8)
                    t2 = (-2*b - sqrt_disc) / (6*a + 1e-8)

                    # Check valid minima
                    second_deriv_t1 = 6*a*t1 + 2*b
                    valid_t1 = (discriminant > 0) & (second_deriv_t1 > 0)
                    t_min = torch.where(valid_t1, t1, t2)
                    valid_min = (discriminant > 0) & ((second_deriv_t1 > 0) | (6*a*t2 + 2*b > 0))

                    # Compute new parameters
                    t_clamped = torch.clamp(t_min, 0, 3)
                    new_param = w0 + a*t_clamped**3 + b*t_clamped**2 + c*t_clamped
                    new_param = torch.where(valid_min, new_param, w3)

                    # Fallback strategies
                    fallback = ~valid_min
                    if fallback.any():
                        # Quadratic fallback using last three points
                        w1_q, w2_q, w3_q = params[1:]
                        a_q = (w3_q - 2*w2_q + w1_q) / 2
                        b_q = (-5*w3_q + 8*w2_q - 3*w1_q) / 2
                        t_q = -b_q / (2*a_q + 1e-8)
                        valid_q = (a_q > 0) & (t_q >= 1) & (t_q <= 3)
                        t_q_clamped = torch.clamp(t_q, 1, 3)
                        new_param_q = w1_q + a_q*(t_q_clamped-1)**2 + b_q*(t_q_clamped-1)
                        new_param[fallback] = torch.where(valid_q[fallback], new_param_q[fallback], w3_q[fallback])

                    p.data.copy_(new_param)
                    state['param_history'].clear()
                    state['grad_history'].clear()
                else:
                    # Standard Adam update if not enough history
                    p.data.add_(-lr * scaled_grad)

        return loss