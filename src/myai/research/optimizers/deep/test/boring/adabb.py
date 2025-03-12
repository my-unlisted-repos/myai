import torch
from torch.optim import Optimizer

class AdaBB(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, gamma=0.99, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, gamma=gamma, eps=eps)
        super(AdaBB, self).__init__(params, defaults)
        
        self.state['global_state'] = {
            'step': 0,
            'x_prev': [],
            'g_prev': [],
            'kappa_avg': 1.0
        }

        # Initialize previous parameters and gradients
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.state['global_state']['x_prev'].append(p.data.clone().detach())
                    self.state['global_state']['g_prev'].append(p.grad.clone().detach())
                else:
                    self.state['global_state']['x_prev'].append(None)
                    self.state['global_state']['g_prev'].append(None)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        global_state = self.state['global_state']
        global_state['step'] += 1

        if global_state['step'] == 1:
            # Save initial parameters and gradients
            new_x_prev, new_g_prev = [], []
            for group in self.param_groups:
                for p in group['params']:
                    new_x_prev.append(p.data.clone().detach() if p.data is not None else None)
                    new_g_prev.append(p.grad.clone().detach() if p.grad is not None else None)
            global_state['x_prev'] = new_x_prev
            global_state['g_prev'] = new_g_prev
            return loss

        # Compute s and y globally
        sy_total = 0.0
        yy_total = 0.0
        ss_total = 0.0
        valid = False

        for i, (group) in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                idx = i * len(group['params']) + j  # Unique index for each parameter
                x_prev = global_state['x_prev'][idx]
                g_prev = global_state['g_prev'][idx]
                if p.grad is None or x_prev is None or g_prev is None:
                    continue
                
                x_current = p.data
                g_current = p.grad.data
                s = x_current - x_prev
                y = g_current - g_prev
                
                sy = torch.dot(s.view(-1), y.view(-1))
                yy = torch.dot(y.view(-1), y.view(-1))
                ss = torch.dot(s.view(-1), s.view(-1))
                
                sy_total += sy.item()
                yy_total += yy.item()
                ss_total += ss.item()
                valid = True

        if not valid or yy_total <= 1e-10 or ss_total <= 1e-10 or sy_total <= 0:
            alpha = self.defaults['lr']
            beta = self.defaults['beta']
        else:
            alpha_BB1 = sy_total / yy_total
            alpha_BB2 = ss_total / sy_total
            kappa = alpha_BB2 / alpha_BB1
            kappa = max(kappa, 1.0)
            
            # Update kappa_avg with exponential moving average
            gamma = self.defaults['gamma']
            global_state['kappa_avg'] = gamma * global_state['kappa_avg'] + (1 - gamma) * kappa
            kappa_avg = global_state['kappa_avg']
            
            sqrt_kappa = torch.sqrt(torch.tensor(kappa_avg))
            beta = ((sqrt_kappa - 1) / (sqrt_kappa + 1)) ** 2
            beta = min(max(beta.item(), 0.0), 0.999)  # Clamp beta between 0 and 0.999
            alpha = alpha_BB1

        # Update parameters with computed alpha and beta
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                # Update momentum buffer
                state['momentum_buffer'].mul_(beta).add_(grad, alpha=alpha)
                # Update parameters
                p.data.add_(-state['momentum_buffer'])

        # Save current parameters and gradients
        new_x_prev, new_g_prev = [], []
        for group in self.param_groups:
            for p in group['params']:
                new_x_prev.append(p.data.clone().detach() if p.data is not None else None)
                new_g_prev.append(p.grad.clone().detach() if p.grad is not None else None)
        global_state['x_prev'] = new_x_prev
        global_state['g_prev'] = new_g_prev

        return loss