import torch
from torch.optim import Optimizer

class ShermanMorrisonAdam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, beta3=0.99, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ShermanMorrisonAdam does not support sparse gradients')

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['u'] = torch.zeros_like(p.data)
                    state['u_ema_bias'] = 0  # Track unnormalized EMA for bias correction

                state['step'] += 1
                m, v, u = state['m'], state['v'], state['u']
                beta1, beta2, beta3 = group['beta1'], group['beta2'], group['beta3']
                eps = group['eps']
                lr = group['lr']

                # Update first moment (m)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update second moment diagonal (v)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Update direction vector (u) with bias correction
                state['u_ema_bias'] = beta3 * state['u_ema_bias'] + (1 - beta3)
                u_bias_correction = 1 - beta3 ** state['step']

                if state['step'] == 1:
                    u.copy_(grad)
                    u_norm = u.norm()
                    if u_norm > 0:
                        u.div_(u_norm + eps)
                else:
                    g_dot_u = torch.sum(grad * u)
                    # Update u: beta3 * u + (1 - beta3) * (g_dot_u * grad)
                    u.mul_(beta3).add_(grad, alpha=(1 - beta3) * g_dot_u)
                    # Normalize u to unit length after bias correction
                    u_hat = u / u_bias_correction
                    u_norm = u_hat.norm()
                    if u_norm > 0:
                        u.copy_(u_hat / (u_norm + eps))

                # Bias correction for m and v
                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # Compute Sherman-Morrison terms with sqrt(v_hat)
                diag_v_inv = 1.0 / (torch.sqrt(v_hat) + eps)
                term1 = m_hat * diag_v_inv

                u_hat = u  # Already normalized and bias-corrected
                u_hat_diag_v_inv = u_hat * diag_v_inv

                numerator = torch.sum(u_hat_diag_v_inv * m_hat)
                denominator = 1 + torch.sum(u_hat * u_hat_diag_v_inv)

                term2 = (u_hat_diag_v_inv * numerator) / denominator

                preconditioned_grad = term1 - term2

                # Update parameters with corrected scaling
                p.data.add_(preconditioned_grad, alpha=-lr)

        return loss

class ShermanMorrisonAdam2(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('CompressedCurvatureOptimizer does not support sparse gradients')

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['u'] = torch.zeros_like(p.data)

                state['step'] += 1
                m, v, u = state['m'], state['v'], state['u']
                beta1, beta2, beta3 = group['betas']

                # Update first moment (gradient EMA)
                m.mul_(beta1).add_(1 - beta1, grad)

                # Update second moment (squared gradient EMA)
                v.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Update sign moment (sign gradient EMA)
                sign_grad = torch.sign(grad)
                u.mul_(beta3).add_(1 - beta3, sign_grad)

                # Bias corrections
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction3 = 1 - beta3 ** state['step']

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                u_hat = u / bias_correction3

                # Compute a = m_hat / (sqrt(v_hat) + eps)
                denom_v = torch.sqrt(v_hat) + group['eps']
                a = m_hat / denom_v

                # Compute b = u_hat / (sqrt(v_hat) + eps)
                b = u_hat / denom_v

                # Flatten tensors for vector operations
                a_flat = a.view(-1)
                u_hat_flat = u_hat.view(-1)
                b_flat = b.view(-1)

                # Compute dot product (a_flat . u_hat_flat)
                dot = torch.dot(a_flat, u_hat_flat)

                # Compute denominator: 1 + (u_hat_flat . b_flat)
                denom = 1 + torch.dot(u_hat_flat, b_flat)

                # Compute correction term
                correction = (dot / denom) * b

                # Preconditioned gradient
                precond_grad = a - correction

                # Update parameters
                p.data.add_(-group['lr'], precond_grad)

        return loss