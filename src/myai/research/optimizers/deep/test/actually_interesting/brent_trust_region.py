import math
from collections import defaultdict, deque

import torch
from torch.optim import Optimizer


class BrentTrustRegion(Optimizer):
    def __init__(self, params, lr=1e-3, history_size=5, trust_radius=1.0,
                 min_trust=1e-6, max_trust=10.0, momentum=0.9, eps=1e-8):
        defaults = dict(lr=lr, history_size=history_size, trust_radius=trust_radius,
                        min_trust=min_trust, max_trust=max_trust, momentum=momentum, eps=eps)
        super().__init__(params, defaults)
        self.state = defaultdict(dict)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['param_history'] = deque(maxlen=history_size + 1)
                state['grad_history'] = deque(maxlen=history_size + 1)
                state['momentum_buffer'] = torch.zeros_like(p.data)
                state['trust_radius'] = trust_radius

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            min_trust = group['min_trust']
            max_trust = group['max_trust']
            momentum = group['momentum']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                state['step'] += 1

                # Store current parameter and gradient
                state['param_history'].append(p.data.clone())
                state['grad_history'].append(grad.clone())

                # Update momentum buffer
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                step_grad = -lr * buf

                # Quasi-Newton step (L-BFGS inspired)
                qn_step = None
                if len(state['param_history']) > 1:
                    s_list, y_list = [], []
                    for i in range(len(state['param_history']) - 1):
                        s = state['param_history'][i+1] - state['param_history'][i]
                        y = state['grad_history'][i+1] - state['grad_history'][i]
                        s_list.append(s)
                        y_list.append(y)
                    qn_step = self.lbfgs_two_loop(grad, s_list, y_list, eps)

                # Choose step
                if qn_step is not None:
                    # Scale to trust region
                    qn_norm = torch.norm(qn_step)
                    trust_radius = state['trust_radius']
                    if qn_norm > trust_radius:
                        qn_step = qn_step * (trust_radius / qn_norm)
                    # Predict reduction
                    grad_dot = torch.dot(grad, qn_step)
                    hessian_term = sum(torch.dot(yi, si) / (torch.dot(si, si) + eps) for si, yi in zip(s_list, y_list))
                    pred_red = grad_dot + 0.5 * hessian_term * grad_dot
                    if pred_red < 0:
                        p.data.add_(qn_step)
                        state['trust_radius'] = min(max_trust, trust_radius * 1.2)
                    else:
                        p.data.add_(step_grad)
                        state['trust_radius'] = max(min_trust, trust_radius * 0.5)
                else:
                    p.data.add_(step_grad)

        return loss

    def lbfgs_two_loop(self, grad, s_list, y_list, eps):
        q = grad.clone()
        alpha_list = []
        for s, y in zip(reversed(s_list), reversed(y_list)):
            rho = 1.0 / (torch.dot(y, s) + eps)
            alpha = rho * torch.dot(s, q)
            q.add_(y, alpha=-alpha)
            alpha_list.append(alpha)
        r = q * (torch.dot(s_list[-1], y_list[-1])) / (torch.dot(y_list[-1], y_list[-1]) + eps)
        for s, y, alpha in zip(s_list, y_list, reversed(alpha_list)):
            rho = 1.0 / (torch.dot(y, s) + eps)
            beta = rho * torch.dot(y, r)
            r.add_(s, alpha=alpha - beta)
        return -r