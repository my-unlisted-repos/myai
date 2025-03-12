import torch
from torch.optim import Optimizer

class BandAdam(Optimizer):
    """Implements a variant of Adam using a banded matrix approximation for the second moment estimate.
    The linear system is solved using preconditioned Conjugate Gradient (CG).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999))
        bandwidth (int, optional): number of off-diagonal bands to maintain (default: 1)
        eps (float, optional): term added to the diagonal to improve numerical stability (default: 1e-8)
        cg_max_iter (int, optional): maximum number of iterations for CG solver (default: 10)
        cg_tol (float, optional): tolerance for convergence in CG solver (default: 1e-5)
        bias_correction (bool, optional): apply bias correction as in Adam (default: True)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), bandwidth=1,
                 eps=1e-8, cg_max_iter=10, cg_tol=1e-5, bias_correction=True):
        defaults = dict(lr=lr, betas=betas, bandwidth=bandwidth,
                        eps=eps, cg_max_iter=cg_max_iter, cg_tol=cg_tol,
                        bias_correction=bias_correction)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('BandAdam does not support sparse gradients')

                state = self.state[p]

                # Initialize state if necessary
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    n = p.numel()
                    bandwidth = group['bandwidth']
                    state['exp_bands'] = []
                    for i in range(bandwidth + 1):
                        size = max(n - i, 0)
                        band = torch.zeros(size, device=p.device)
                        state['exp_bands'].append(band)

                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_bands = state['exp_bands']
                beta1, beta2 = group['betas']
                grad_flat = grad.view(-1)
                n = grad_flat.size(0)
                bandwidth = group['bandwidth']

                # Update first moment
                exp_avg.mul_(beta1).add_(grad_flat, alpha=1 - beta1)

                # Update bands for second moment
                # Main diagonal
                exp_bands[0].mul_(beta2).add_(grad_flat.pow(2), alpha=1 - beta2)
                # Upper diagonals
                for i in range(1, bandwidth + 1):
                    if i >= n:
                        continue
                    band_grad = grad_flat[:-i] * grad_flat[i:]
                    exp_bands[i].mul_(beta2).add_(band_grad, alpha=1 - beta2)

                # Bias correction
                step = state['step']
                bias_correction1 = 1.0
                bias_correction2 = 1.0
                if group['bias_correction']:
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                m_flat = exp_avg / bias_correction1
                main_band = exp_bands[0] / bias_correction2 + group['eps']
                upper_bands = [band / bias_correction2 for band in exp_bands[1:]]

                # Solve Mx = m_flat using preconditioned CG
                x = self._conjugate_gradient_solve(
                    main_band=main_band,
                    upper_bands=upper_bands,
                    b=m_flat,
                    max_iter=group['cg_max_iter'],
                    tol=group['cg_tol']
                )

                # Update parameters
                p.data.add_(x.view_as(p.data), alpha=-group['lr'])

        return loss

    def _conjugate_gradient_solve(self, main_band, upper_bands, b, max_iter, tol):
        """Solves the linear system Mx = b using preconditioned CG."""
        n = main_band.size(0)
        x = torch.zeros_like(b)
        r = b - self._banded_matvec(main_band, upper_bands, x)
        z = r / main_band  # Diagonal preconditioner
        p = z.clone()
        rsold = torch.dot(r, z)

        for _ in range(max_iter):
            Ap = self._banded_matvec(main_band, upper_bands, p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-10)  # Avoid division by zero
            x.add_(p, alpha=alpha)
            r.sub_(Ap, alpha=alpha)
            if torch.norm(r) < tol:
                break
            z = r / main_band
            rsnew = torch.dot(r, z)
            beta = rsnew / (rsold + 1e-10)
            p = z + beta * p
            rsold = rsnew

        return x

    def _banded_matvec(self, main_band, upper_bands, v):
        """Computes M*v where M is a symmetric banded matrix."""
        result = main_band * v
        for i, band in enumerate(upper_bands):
            d = i + 1
            if d >= v.size(0):
                continue
            result[:-d] += band * v[d:]
            result[d:] += band * v[:-d]
        return result