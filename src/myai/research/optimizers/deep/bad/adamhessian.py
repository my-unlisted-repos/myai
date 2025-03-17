import torch
from torch.optim import Optimizer

class AdamHessian(Optimizer):
    """very unstable and basically useless"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, normalize=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, normalize=normalize)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                # Initialize state if not present
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['h_ema'] = torch.zeros_like(p)
                    state['theta_prev'] = p.clone()
                    state['g_prev'] = grad.clone()
                    p.sub_(p.grad * lr*1e-3)
                else:
                    # Compute s and y for Hessian estimate
                    theta_prev_current = p.clone()
                    s = theta_prev_current - state['theta_prev']
                    y = grad - state['g_prev']

                    # Avoid division by zero by using a small epsilon
                    h_t = y / (s + eps)

                    # Update Hessian EMA
                    state['h_ema'] = torch.addcmul(
                        state['h_ema'] * beta2,
                        h_t,
                        torch.tensor(1 - beta2, dtype=torch.float32, device=p.device)
                    )

                    # Update first moment (gradient EMA)
                    state['m'] = torch.addcmul(
                        state['m'] * beta1,
                        grad,
                        torch.tensor(1 - beta1, dtype=torch.float32, device=p.device)
                    )

                    state['step'] += 1

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    m_hat = state['m'] / bias_correction1
                    h_ema_hat = state['h_ema'] / bias_correction2

                    # Compute update
                    denom = h_ema_hat + eps
                    update = m_hat / denom

                    # Apply update
                    if group['normalize']: update /= (torch.linalg.vector_norm(update) + 1e-8)
                    p.add_(-lr * update)

                    # Save current parameters and gradients for next step
                    state['theta_prev'].copy_(theta_prev_current)
                    state['g_prev'].copy_(grad)

        return loss