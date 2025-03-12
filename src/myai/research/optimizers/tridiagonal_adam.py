"""Adam but it uses a little bit more of the covariance matrix"""
from typing import Literal
import torch
from torch.optim import Optimizer

def _matrix_vector_product_tridiag(diag, offdiag, x):
    """Computes A @ x for a symmetric tridiagonal matrix A."""
    Ax = diag * x
    if len(diag) > 1:
        Ax[:-1] += offdiag * x[1:]
        Ax[1:] += offdiag * x[:-1]
    return Ax


def solve_tridiagonal_cg(diag:torch.Tensor, off_diag:torch.Tensor, v:torch.Tensor, x_init=None, iters=None, tol=1e-6, jacobi=True):
    """
    solves Ax = v for a symmetric tridiagonal matrix A with conjugate gradient.

    Args:
        diag (torch.Tensor): diagonal vector of A (shape N).
        off_diag (torch.Tensor): off-diagonal vector of A (shape N-1).
        v (torch.Tensor): right-hand side vector (shape N).
        x_init (torch.Tensor, optional): initial guess for the solution x. Defaults to None (zeros).
        iters (int | None, optional): maximum number of iterations (defaults to N).
        tol (float): tolerance for convergence (stopping criterion based on residual norm).
        jacobi (bool): whether to use jacobi preconditioning (it helps a little).

    Returns:
        torch.Tensor: Approximated solution x.
    """
    N = diag.size(0)
    if iters is None: iters = N

    if x_init is None: x = torch.zeros_like(v)
    else: x = x_init

    # initial residual: r = v - Ax
    r = v - _matrix_vector_product_tridiag(diag, off_diag, x)
    p = r.clone()  # initial conjugate direction p = r
    rho = torch.dot(r, r)


    if jacobi:
        M_inv = 1.0 / (diag + 1e-8)
        z = M_inv * r
    else:
        z = None
        M_inv = None

    success = True
    iters_used = 0
    for i in range(iters):
        Ap = _matrix_vector_product_tridiag(diag, off_diag, p)
        alpha = rho / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        if jacobi:
            assert M_inv is not None and z is not None
            z_new = M_inv * r
        else:
            z_new = r

        rho_new = torch.dot(r, z_new)

        if torch.sqrt(rho_new) < tol: # convergence check
            break

        beta = rho_new / rho
        p = z_new + beta * p
        rho = rho_new

        iters_used += 1

    else:
        # tolerance condition not met
        success = False

    return x, success, iters_used


def solve_tridiagonal_jacobi(diag:torch.Tensor, off_diag:torch.Tensor, v:torch.Tensor, x_init=None, iters=100, tol = None):
    """
    Solves Ax = v for a symmetric tridiagonal matrix A using Jacobi iteration, this requires way more iters than CG,
    now I also thought using solution from previous step would help but it doesn't.

    Args:
        diag (torch.Tensor): diagonal vector of A (shape N).
        off_diag (torch.Tensor): off-diagonal vector of A (shape N-1).
        v (torch.Tensor): right-hand side vector (shape N).
        x_init (torch.Tensor, optional): initial guess for the solution x. Defaults to None (zeros).
        iters (int): number of Jacobi iterations.
        tol: unused for this one but all solvers have same args.

    Returns:
        torch.Tensor: Approximate solution x (shape N).
    """

    if x_init is None:
        x = torch.zeros_like(v)
    else:
        x = x_init

    diag_inv = 1.0 / diag

    for _ in range(iters):
        x_new = torch.zeros_like(v)

        x_new[0] = (v[0] - off_diag[0] * x[1]) * diag_inv[0]
        x_new[1:-1] = (v[1:-1] - off_diag[:-1] * x[:-2] - off_diag[1:] * x[2:]) * diag_inv[1:-1]
        x_new[-1] = (v[-1] - off_diag[-1] * x[-2]) * diag_inv[-1]

        x = x_new

    return x, True, iters


class TridiagonalAdam(Optimizer):
    """adam with tridiagonal covariance matrix but it doesn't use square root.

    Args:
        params (_type_): params to optimize
        lr (_type_, optional): learning rate. Defaults to 1e-3.
        betas (tuple, optional):
            beta1 is momentum and beta2 is covariance matrix tridiagonal EMA. Defaults to (0.9, 0.999).
        eps (_type_, optional): division eps. Defaults to 1e-8.
        weight_decay (int, optional): not decoupled (for now). Defaults to 0.
        use_init (bool, optional):
            whether to pass solution of previous step Ax=g as initial x for the next step, needs testing. Defaults to False.
        tol (_type_, optional): tolerance for Ax=g solver. Defaults to 1e-6.
        iters (int, optional): number of Ax=g solver iterations. Defaults to 5.
        solver (_type_, optional): solver. Defaults to solve_tridiagonal_cg.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, use_init = False, tol=1e-6, iters=6, solver = solve_tridiagonal_cg):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, tol=tol, iters=iters, use_init = use_init, solver = solver)
        super().__init__(params, defaults)

        self._n_success = 0
        self._n_fail = 0
        self._n_solver_iters = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TridiagonalAdam does not support sparse gradients")
                state = self.state[p]

                # initialize
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['diag_ema'] = torch.ones_like(p, memory_format=torch.preserve_format)
                    state['off_diag_ema'] = torch.zeros_like(p.view(-1)[:-1], memory_format=torch.preserve_format)

                exp_avg, diag_ema, off_diag_ema = state['exp_avg'], state['diag_ema'], state['off_diag_ema']
                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # diagonal ema
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                diag_ema.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # off diagonal ema
                g = grad.view(-1)
                grad_offdiag = g[:-1] * g[1:]
                off_diag_ema.mul_(beta2).add_(grad_offdiag, alpha=1 - beta2)

                # adam stuff
                denom_diag_hat = diag_ema.div(bias_correction2)
                denom_offdiag_hat = off_diag_ema.div(bias_correction2)
                m_hat = exp_avg.div(bias_correction1)

                # solve Ax = g
                use_init = group['use_init']
                if use_init and 'prev_solution' in state:
                    x_init = state['prev_solution']
                else:
                    x_init = None

                solver = group['solver']

                update_direction, success, iters = solver(
                    denom_diag_hat.view(-1),
                    denom_offdiag_hat,
                    m_hat.view(-1),
                    x_init=x_init,
                    tol=group["tol"],
                    iters=group["iters"],
                )
                # log those to test if jacobi preconditioning and use_init help
                self._n_solver_iters += iters
                if success: self._n_success += 1
                else: self._n_fail += 1

                if use_init:
                    state['prev_solution'] = update_direction

                p.add_(update_direction.view_as(p), alpha=-group['lr'])


        return loss