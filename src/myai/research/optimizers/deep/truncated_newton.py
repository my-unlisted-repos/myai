import torch
from torch.optim import Optimizer

class TruncatedNewton(Optimizer):
    def __init__(self, params, lr=0.1, cg_iters=10, epsilon=1e-5):
        defaults = dict(lr=lr, cg_iters=cg_iters, epsilon=epsilon)
        super().__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise ValueError("Closure required for TruncatedNewton")

        # Compute initial loss and gradients
        loss = closure()
        # self.zero_grad()
        # loss.backward()

        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                grads.append(p.grad.detach().clone())

        # Flatten the gradients
        g_flat = torch.cat([g.reshape(-1) for g in grads])

        # Right-hand side of Hx = -g
        b = -g_flat

        # Initial CG variables
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rsold = torch.dot(r, r)

        cg_iters = self.defaults['cg_iters']
        epsilon = self.defaults['epsilon']

        for _ in range(cg_iters):
            Hp = self._compute_Hp(params, p, closure, epsilon)

            Hp_dot_p = torch.dot(p, Hp)
            if Hp_dot_p.abs() < 1e-10:
                break
            alpha = rsold / Hp_dot_p

            x += alpha * p
            r -= alpha * Hp
            rsnew = torch.dot(r, r)

            if rsnew.sqrt() < 1e-6:
                break

            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew

        # Update parameters
        self._update_params(params, x, self.defaults['lr'])

    def _compute_Hp(self, params, p_vec, closure, epsilon):
        orig_params = [param.detach().clone() for param in params]
        orig_grads = [param.grad.detach().clone() for param in params]

        # Perturb parameters by epsilon * p_vec
        offset = 0
        with torch.no_grad():
            for param in params:
                numel = param.numel()
                p_part = p_vec[offset:offset + numel].view_as(param)
                param.add_(epsilon * p_part)
                offset += numel

        # Compute perturbed gradients
        # self.zero_grad()
        loss_perturbed = closure()
        # loss_perturbed.backward()
        perturbed_grads = [param.grad.detach().clone() for param in params]

        # Compute Hv
        Hp = []
        for orig_grad, pert_grad in zip(orig_grads, perturbed_grads):
            Hp.append((pert_grad - orig_grad) / epsilon)
        Hp_flat = torch.cat([h.reshape(-1) for h in Hp])

        # Restore parameters and gradients
        with torch.no_grad():
            for param, orig_param in zip(params, orig_params):
                param.copy_(orig_param)
        for param, orig_grad in zip(params, orig_grads):
            if param.grad is not None:
                param.grad.copy_(orig_grad)

        return Hp_flat

    def _update_params(self, params, direction, lr):
        offset = 0
        with torch.no_grad():
            for param in params:
                numel = param.numel()
                update = direction[offset:offset + numel].view_as(param)
                param.add_(lr * update)
                offset += numel