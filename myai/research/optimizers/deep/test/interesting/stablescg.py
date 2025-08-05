# pylint:disable=signature-differs, not-callable # type:ignore
import torch

class StableSCG(torch.optim.Optimizer):
    """Conjugate gradient that can work with mini-batch and it is quite stable, requires 2 closure evaluations per batch and conjugate direction is formed on the same batch, while this sounds bad it can still be faster than Kron or SOAP. even despite not being foreach."""
    def __init__(self, params, lr=1e-1, delta=1e-4,
                 restart_freq=10, beta_momentum=0.9,
                 max_alpha=100.0, grad_smooth=0.9):
        defaults = dict(lr=lr, delta=delta,
                        restart_freq=restart_freq,
                        beta_momentum=beta_momentum,
                        max_alpha=max_alpha,
                        grad_smooth=grad_smooth)
        super().__init__(params, defaults)
        self.state.setdefault('step', 0)

        # Initialize state buffers
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['d_prev'] = torch.zeros_like(p)
                state['g_avg'] = torch.zeros_like(p)


        self.num_cg_steps = 0
        self.num_gd_steps = 0

    @torch.no_grad
    def step(self, closure):
        with torch.enable_grad(): loss = closure()

        if len(self.param_groups) != 1:
            raise NotImplementedError(f"StableSCG only supports 1 parameter group, got {len(self.param_groups)}")

        for group in self.param_groups:
            lr = group['lr']
            delta = group['delta']
            max_alpha = group['max_alpha']
            grad_smooth = group['grad_smooth']

            params = []
            g_list = []
            d_prev_list = []
            g_avg_list = []

            # Gather parameters and gradients
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                params.append(p)
                g_list.append(p.grad.detach().clone())
                d_prev_list.append(state['d_prev'])
                g_avg_list.append(state['g_avg'])

            if not params:
                continue

            # Smooth gradients
            for g_avg, g_new in zip(g_avg_list, g_list):
                g_avg.mul_(grad_smooth).add_(g_new, alpha=1-grad_smooth)

            # Compute beta using Polak-Ribière
            step = self.state['step']
            if step == 0 or step % group['restart_freq'] == 0:
                beta = 0.0
            else:
                pr_numerator = 0.0
                g_old_norm = 0.0
                for g_avg_prev, g_avg in zip(self.g_avg_list_prev, g_avg_list): # pylint:disable=access-member-before-definition
                    pr_numerator += torch.dot(
                        (g_avg - g_avg_prev).flatten(),
                        g_avg.flatten()
                    ).item()
                    g_old_norm += torch.dot(
                        g_avg_prev.flatten(),
                        g_avg_prev.flatten()
                    ).item()

                beta = pr_numerator / (g_old_norm + 1e-12)
                beta = max(min(beta, 0.9/(1-group['beta_momentum'])), -0.9/(1-group['beta_momentum']))

            # Compute search direction
            d_list = []
            for g_avg, d_prev in zip(g_avg_list, d_prev_list):
                d = -g_avg + beta * d_prev
                d_list.append(d)

            # ---- Curvature estimation ----
            # Save original parameters
            orig_params = [p.detach().clone() for p in params]

            # Perturb parameters
            for p, d in zip(params, d_list):
                p.add_(delta * d)

            # Compute perturbed gradients
            with torch.enable_grad(): loss_perturbed = closure()
            g_perturbed = []
            for p in params:
                if p.grad is not None:
                    g_perturbed.append(p.grad.detach().clone())
                    p.grad.detach_()
                    p.grad.zero_()

            # Restore parameters
            for p, orig in zip(params, orig_params):
                p.copy_(orig)

            # Compute curvature
            curvature = 0.0
            for g, g_p, d in zip(g_avg_list, g_perturbed, d_list):
                hv = (g_p - g) / delta
                curvature += torch.dot(hv.flatten(), d.flatten()).item()

            # ---- Step size calculation ----
            g_norm_sq = sum(torch.dot(g.flatten(), g.flatten()).item()
                           for g in g_avg_list)

            if curvature <= 1e-12:
                # Negative or near-zero curvature - use gradient descent
                self.num_gd_steps += 1
                alpha = lr
                d_list = [-g for g in g_avg_list]  # Reset to GD direction
            else:
                self.num_cg_steps += 1
                alpha_raw = g_norm_sq / (abs(curvature) + 1e-12)
                alpha = min(alpha_raw * lr, max_alpha)
                if curvature < 0:
                    alpha *= -1  # Follow negative curvature direction

            # Parameter update
            for p, d in zip(params, d_list):
                p.add_(alpha * d)

            # Store state for next iteration
            for p, g_avg, d in zip(params, g_avg_list, d_list):
                state = self.state[p]
                state['d_prev'] = d.clone()
                state['g_avg_prev'] = g_avg.clone()

            self.g_avg_list_prev = g_avg_list  # Store for next beta calculation
            self.state['step'] += 1

        return loss