# pylint:disable=signature-differs, not-callable # type:ignore

import torch
from torch.optim import Optimizer
from torchzero.utils.torch_tools import swap_tensors_no_use_count_check


def vector_add(v1, v2, alpha=1.0):
    return [t1 + alpha * t2 for t1, t2 in zip(v1, v2)]

def vector_mul(v, alpha):
    return [t * alpha for t in v]

def vector_dot(v1, v2):
    return sum(torch.sum(t1 * t2) for t1, t2 in zip(v1, v2))

def vector_norm(v):
    return torch.sqrt(vector_dot(v, v))

def vector_neg(v):
    return [-t for t in v]

def conjugate_gradient(hvp_fn, b, max_iter=50, tol=1e-10):
    x = [torch.zeros_like(t) for t in b]
    r = b  # r = b (since x is 0)
    p = r.copy()
    rsold = vector_dot(r, r)
    for i in range(max_iter):
        Hp = hvp_fn(p)
        alpha = rsold / (vector_dot(p, Hp) + 1e-8)
        x = vector_add(x, p, alpha)
        r = vector_add(r, Hp, -alpha)
        rsnew = vector_dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        beta = rsnew / rsold
        p = vector_add(r, p, beta)
        rsold = rsnew
    return x

class TrustRegionSearch(Optimizer):
    """similar to line search but instead it is dogleg trust region search"""
    def __init__(self, params, lr=1.0, max_trust_radius=100.0, initial_trust_radius=0.1, cg_max_iter=50, cg_tol=1e-5):
        defaults = dict(lr=lr, max_trust_radius=max_trust_radius,
                        initial_trust_radius=initial_trust_radius,
                        cg_max_iter=cg_max_iter, cg_tol=cg_tol)
        super().__init__(params, defaults)
        self.state['trust_radius'] = initial_trust_radius


    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for TrustRegionDogleg")

        assert len(self.param_groups) == 1
        # Initial closure call to compute loss and gradients
        with torch.enable_grad():
            self.zero_grad()
            loss = init_loss = closure(False)
            loss.backward(create_graph=True)
            grads = []
            for group in self.param_groups:
                for p in group['params']:
                    # if p.grad is None:
                    #     raise ValueError("Gradients not found in closure")
                    grads.append(p.grad.clone())

            # Compute HVP function
            params = self.param_groups[0]['params']
            def hvp_fn(v):
                hv = torch.autograd.grad(
                    grads,
                    params,
                    grad_outputs=v,
                    retain_graph=True,
                    allow_unused=True,
                )
                hv = [h if h is not None else torch.zeros_like(p) for h, p in zip(hv, params)]
                return hv

            # Compute Newton direction using CG
            b = vector_neg(grads)
            p_newton = conjugate_gradient(hvp_fn, b, self.defaults['cg_max_iter'], self.defaults['cg_tol'])

            # Compute Cauchy step
            g = grads
            Hg = hvp_fn(g)
        g_norm = vector_norm(g)
        g_H_g = vector_dot(g, Hg)
        trust_radius = self.state['trust_radius']

        if g_H_g <= 0:
            alpha_c = trust_radius / g_norm
        else:
            alpha_c = min((g_norm ** 2) / g_H_g, trust_radius / g_norm)
        p_cauchy = vector_mul(g, -alpha_c)

        # Compute candidate deltas
        candidate_deltas = [0.5*trust_radius, trust_radius, 2*trust_radius]
        best_rho = -float('inf')
        best_step = None
        best_loss = loss


        for delta in candidate_deltas:
            # Compute Dogleg step
            p_cauchy_norm = vector_norm(p_cauchy)
            p_newton_norm = vector_norm(p_newton)

            if p_newton_norm <= delta:
                step = p_newton
            elif p_cauchy_norm >= delta:
                step = vector_mul(p_cauchy, delta / p_cauchy_norm)
            else:
                d = vector_add(p_newton, p_cauchy, -1.0)
                a = vector_dot(d, d)
                b_val = 2 * vector_dot(p_cauchy, d)
                c_val = vector_dot(p_cauchy, p_cauchy) - delta ** 2
                discriminant = b_val ** 2 - 4 * a * c_val
                t = (-b_val + torch.sqrt(discriminant)) / (2 * a)
                step = vector_add(p_cauchy, d, t)

            # Temporarily apply step and compute loss
            with torch.no_grad():
                other = [(p+s) for p,s in zip(params,step)]
                for p, o in zip(params, other):
                    swap_tensors_no_use_count_check(p, o)
                new_loss = closure(False)
                for p, o in zip(params, other):
                    swap_tensors_no_use_count_check(o, p)

            # Compute predicted reduction
            g_dot_step = vector_dot(g, step)
            if vector_norm(step) == p_newton_norm:
                # Newton step
                p_H_p = vector_dot(step, vector_neg(g))
            elif vector_norm(step) == delta and vector_norm(p_cauchy) >= delta:
                # Scaled Cauchy step
                p_H_p = (delta ** 2 / (alpha_c ** 2)) * vector_dot(p_cauchy, Hg) * alpha_c ** 2
            else:
                # Line segment step (requires precomputed terms, simplified here)
                H_step = hvp_fn(step)
                p_H_p = vector_dot(step, H_step)

            predicted_reduction = - (g_dot_step + 0.5 * p_H_p)
            actual_reduction = loss - new_loss
            rho = actual_reduction / (predicted_reduction + 1e-8)

            if rho > best_rho:
                best_rho = rho
                best_step = step
                best_loss = new_loss

        # Update parameters and adjust trust region
        if best_rho > 0:
            with torch.no_grad():
                for p, s in zip(params, best_step):
                    p.add_(s)
        else:
            best_loss = init_loss

        # Adjust trust region
        if best_rho < 0.25:
            self.state['trust_radius'] *= 0.25
        elif best_rho > 0.75:
            self.state['trust_radius'] = min(2 * self.state['trust_radius'], self.defaults['max_trust_radius'])

        return best_loss