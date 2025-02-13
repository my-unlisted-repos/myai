# pylint:disable=signature-differs, not-callable

import torch
from torch.optim import Optimizer

class CubicRegularizedNewton(Optimizer):
    def __init__(self, params, lr=1.0, sigma=1.0, max_iter=5):
        defaults = dict(lr=lr, sigma=sigma, max_iter=max_iter)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        # Make sure we have a closure
        closure = torch.enable_grad()(closure)

        # Get initial loss and gradients
        with torch.enable_grad():
            loss = closure(False)

            # Zero gradients before backward pass
            self.zero_grad()
            # Compute first-order gradients with create_graph for Hessian
            loss.backward(create_graph=True)

            # Flatten all parameters and gradients
            params = []
            grads = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        params.append(p)
                        grads.append(p.grad.view(-1))

            flat_grad = torch.cat(grads).detach()
            flat_params = torch.cat([p.view(-1).detach() for p in params])

            # Compute Hessian (Jacobian of gradients)
            hessian_rows = []
            for i in range(flat_grad.size(0)):
                # Compute gradient of i-th gradient component
                grad_i = grads[0][i] if len(params) == 1 else torch.cat([g[i:i+1] for g in grads])
                hessian_row = torch.autograd.grad(
                    grad_i, params,
                    retain_graph=(i < flat_grad.size(0)-1),
                    allow_unused=True
                )

                # Handle parameters with no gradient
                processed_row = []
                for p, hr in zip(params, hessian_row):
                    if hr is None:
                        processed_row.append(torch.zeros_like(p.view(-1)))
                    else:
                        processed_row.append(hr.contiguous().view(-1))

                hessian_rows.append(torch.cat(processed_row))

        hessian = torch.stack(hessian_rows).detach()
        n = hessian.size(0)

        # Cubic regularization solve: (H + σ||d||I)d = -g
        d = -flat_grad  # Initial guess
        sigma = self.param_groups[0]['sigma']
        for _ in range(self.param_groups[0]['max_iter']):
            d_norm = torch.norm(d)
            H_reg = hessian + sigma * d_norm * torch.eye(n, device=hessian.device)

            try:
                d = torch.linalg.solve(H_reg, -flat_grad)
            except torch.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                d = -torch.linalg.pinv(H_reg) @ flat_grad

        # Update parameters
        index = 0
        for p in params:
            size = p.numel()
            p.data.add_(d[index:index+size].view_as(p.data),
                       alpha=self.param_groups[0]['lr'])
            index += size

        return loss