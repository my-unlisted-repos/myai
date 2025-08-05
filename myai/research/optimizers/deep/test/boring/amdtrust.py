import torch
import math

class AMDTrust(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, beta3=0.9,
                 eps=1e-8, trust_param=0.1, var_param=0.1):
        """
        AMDTrust: Adaptive Momentum with Dynamic Trust Optimizer
        - Combines adaptive learning rates with momentum adjusted by gradient direction trust.
        - Incorporates variance penalization to stabilize steps during oscillations.
        - Args:
            lr: Learning rate (default: 1e-3)
            beta1: Momentum decay (default: 0.9)
            beta2: Second moment decay (default: 0.999)
            beta3: Variance term decay (default: 0.9)
            eps: Numerical stability (default: 1e-8)
            trust_param: Momentum boost factor for consistent directions (default: 0.1)
            var_param: Variance penalty strength (default: 0.1)
        """
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta3=beta3,
                       eps=eps, trust_param=trust_param, var_param=var_param)
        super(AMDTrust, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)  # First moment
                    state['v'] = torch.zeros_like(p.data)  # Second moment
                    state['var'] = torch.zeros_like(p.data)  # Variance term
                    state['g_prev'] = torch.zeros_like(p.data)  # Previous gradient

                m, v, var, g_prev = state['m'], state['v'], state['var'], state['g_prev']
                beta1 = group['beta1']
                beta2 = group['beta2']
                beta3 = group['beta3']
                eps = group['eps']
                trust_param = group['trust_param']
                var_param = group['var_param']

                # Compute cosine similarity between gradients
                g_prev_norm = torch.norm(g_prev)
                grad_norm = torch.norm(grad)
                if g_prev_norm == 0 or grad_norm == 0:
                    cos_sim = 0.0
                else:
                    cos_sim = torch.dot(grad.view(-1), g_prev.view(-1)) / (grad_norm * g_prev_norm)
                trust = 0.5 * (cos_sim + 1)  # [0, 1] trust score

                # Compute variance term (gradient change)
                delta_g = grad - g_prev
                var_t = beta3 * var + (1 - beta3) * (delta_g ** 2)
                var_scaled = var_t.sqrt()  # sqrt(variance)

                # Adjust momentum based on trust
                beta1_t = beta1 * trust + (1 - trust) * 0.5  # beta1_min = 0.5

                # Update moments
                m.mul_(beta1_t).add_(grad, alpha=1 - beta1_t)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute denominator (combines second moment and variance)
                denom = (v + var_t).sqrt().add_(eps)

                # Adaptive learning rate with bias correction
                bias_correction2 = 1 - beta2 ** (state['step'] + 1)
                lr_t = group['lr'] * math.sqrt(bias_correction2) / denom

                # Variance penalty: reduce step size when variance is high
                lr_t.div_(1 + var_param * var_scaled)

                # Compute update step
                delta = lr_t * m

                # Boost step size when trust is high (consistent direction)
                delta.mul_(1 + trust_param * trust)

                # Apply update
                p.data.add_(-delta)

                # Update state variables
                state['step'] += 1
                state['m'], state['v'], state['var'], state['g_prev'] = m, v, var_t, grad.clone()

        return loss