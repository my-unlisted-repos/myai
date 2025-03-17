from collections import deque

import torch
from torch.optim import Optimizer


def conjugate_gradient(matvec, b, max_iter=50, x0 = None, tol=1e-6, eps=1e-8, preconditioner=None):
    """CG + diagonal preconditioning"""
    x = torch.zeros_like(b) if x0 is None else x0
    r = b.clone()
    if preconditioner is None:
        p = r.clone()
    else:
        p = preconditioner(r)
    rsold = torch.dot(r, p)

    for i in range(max_iter):
        Ap = matvec(p)
        alpha = rsold / (torch.dot(p, Ap) + eps)
        x = x + alpha * p
        r = r - alpha * Ap

        if preconditioner is not None:
            z = preconditioner(r)
        else:
            z = r.clone()

        rsnew = torch.dot(r, z)
        if torch.sqrt(rsnew) < tol:
            break
        p = z + (rsnew / (rsold + eps)) * p
        rsold = rsnew

    return x

class EigenPreconditionedOptimizer(Optimizer):
    """stores a history of eigenvalues and eigenvectors of the covariance matrix and uses them to approximate FIM,
    this doesn't use square root. It is good but slow due to cg and sometimes unstable."""
    def __init__(self, params, lr=1e-3, beta = 0.95, history_size=10,
                 eps=1e-6, cg_max_iters=5, damping=1e-7, use_sqrt=False, max_norm_growth = None, trust_min = 0.1, trust_max=10):
        defaults = dict(
            lr=lr,
            history_size=history_size,
            eps=eps,
            cg_max_iters=cg_max_iters,
            damping=damping,
            use_sqrt=use_sqrt,
            beta = beta,
            max_norm_growth = max_norm_growth,
            trust_min = trust_min,
            trust_max = trust_max,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['history'] = deque(maxlen=group['history_size'])
                state['diag_approx'] = torch.zeros_like(p.data)  # Diagonal preconditioner

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            damping = group['damping']
            cg_max_iters = group['cg_max_iters']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if 'step' not in state: state['step'] = 1
                else: state['step'] += 1

                # Update diagonal preconditioner estimate
                state['diag_approx'].mul_(beta).add_(grad.pow(2), alpha=1 - beta)

                # Update eigen history
                g_norm = torch.norm(grad)
                if g_norm > 1e-8:
                    eigenvec = (grad / g_norm).detach()
                    eigenval = g_norm.pow(2).detach()
                    state['history'].append((eigenval, eigenvec))

                # Build matrix-vector product operator
                diag_precond = 1 / (state['diag_approx'].sqrt() + eps)
                history = list(state['history'])

                def matvec(x_flat):
                    x = x_flat.view_as(p)
                    result = damping * x.clone()  # Damping term
                    if group['use_sqrt']: result.addcmul_(state['diag_approx'].sqrt(), x)  # Diagonal preconditioning
                    else: result.addcmul_(state['diag_approx'], x)
                    for lam, v in history:
                        result.add_(lam * v * torch.sum(v * x))
                    return result.flatten()

                # Solve using preconditioned CG
                b = grad.flatten()
                x_flat = conjugate_gradient(
                    matvec=matvec,
                    b=b,
                    max_iter=cg_max_iters,
                    eps=eps,
                    preconditioner=lambda r: diag_precond.flatten() * r
                )

                # Adaptive step size scaling
                update = x_flat.view_as(p)
                update_norm = torch.norm(update)
                grad_norm = torch.norm(grad)

                if group['max_norm_growth'] is not None:
                    if 'prev_norm' not in state: state['prev_norm'] = update_norm
                    else:
                        prev_norm = state['prev_norm']
                        if update_norm / prev_norm > group['max_norm_growth']:
                            update.div_((update_norm / prev_norm) * (1 / group['max_norm_growth']))
                            prev_norm = prev_norm * group['max_norm_growth']
                        else:
                            prev_norm = max(update_norm, prev_norm / group['max_norm_growth'], 1e-4)
                        state['prev_norm'] = prev_norm

                if state['step'] < group['history_size']:
                    warmup_lr = lr / (group['history_size'] - state['step'])
                else:
                    warmup_lr = lr
                if update_norm > 0 and grad_norm > 0:
                    trust_ratio = torch.clamp(grad_norm / update_norm, group['trust_min'], group['trust_max'])
                    p.add_(update, alpha=-warmup_lr * trust_ratio)

        return loss