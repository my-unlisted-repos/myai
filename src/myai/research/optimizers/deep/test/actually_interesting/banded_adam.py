import torch
from torch.optim import Optimizer

def banded_matrix_vector_multiply(banded_matrix_bands, vector, bandwidth):
    """
    Multiplies a banded matrix (stored as bands) by a vector.

    Args:
        banded_matrix_bands: List of tensors representing the bands of the matrix.
                             banded_matrix_bands[0] is the main diagonal,
                             banded_matrix_bands[1] is the first upper/lower diagonal, and so on.
        vector: The vector to multiply by (torch.Tensor).
        bandwidth: The bandwidth of the banded matrix.

    Returns:
        The result of the banded matrix-vector multiplication (torch.Tensor).
    """
    n = vector.size(0)
    result = torch.zeros_like(vector)

    # Main diagonal
    result += banded_matrix_bands[0] * vector

    # Upper and lower diagonals
    for i in range(1, bandwidth + 1):
        if i < n: # handle edge cases near matrix boundaries.
            upper_diag = banded_matrix_bands[i]
            lower_diag = banded_matrix_bands[i] # Assuming symmetric for simplicity, adjust if needed.

            # Upper diagonal contribution
            result[i:] += upper_diag[:n-i] * vector[:-i]

            # Lower diagonal contribution
            result[:-i] += lower_diag[:n-i] * vector[i:]
    return result


def cg(A_multiply, b, x0=None, max_iter=None, tolerance=1e-6, preconditioner=None):
    """
    Conjugate Gradient (CG) solver for Ax = b.
    A_multiply should be a function that takes a vector and returns A*vector.
    Here, A_multiply can be for banded or dense matrix.

    Args:
        A_multiply: Function to compute matrix-vector product (callable).
        b: Right-hand side vector (torch.Tensor).
        x0: Initial guess for the solution (torch.Tensor, optional).
        max_iter: Maximum number of iterations (int, optional).
        tolerance: Convergence tolerance (float).
        preconditioner: Preconditioner function M_inv_multiply, solves M^{-1}r (callable, optional).

    Returns:
        Tuple: (solution x, number of iterations).
    """
    n = b.size(0)
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
    r = b - A_multiply(x)
    if preconditioner is not None:
        z = preconditioner(r)
        p = z.clone()
    else:
        p = r.clone()
    r_dot_z_new = torch.dot(r, z) if preconditioner else torch.dot(r, r)
    r_dot_z_initial = r_dot_z_new
    iteration = 0

    if max_iter is None:
        max_iter = n  # Maximum possible iterations without preconditioning

    while iteration < max_iter and r_dot_z_new > tolerance**2 * r_dot_z_initial:
        Ap = A_multiply(p)
        alpha = r_dot_z_new / torch.dot(p, Ap)
        x.add_(alpha * p)
        r.subtract_(alpha * Ap)
        if preconditioner is not None:
            z = preconditioner(r)
            r_dot_z_old = r_dot_z_new
            r_dot_z_new = torch.dot(r, z)
            beta = r_dot_z_new / r_dot_z_old
            p = z + beta * p
        else:
            r_dot_r_old = r_dot_z_new # reusing variable for clarity when no preconditioner
            r_dot_z_new = torch.dot(r, r) # when no preconditioner, z = r
            beta = r_dot_z_new / r_dot_r_old
            p = r + beta * p
        iteration += 1
    return x, iteration


class BandedAdam(Optimizer):
    """
    Banded Adam optimizer.

    Implements Adam with a banded or full covariance approximation of the second moment matrix
    and Conjugate Gradient for solving the update linear system.
    Uses shifted gradient products for banded covariance estimation and
    full covariance for parameters smaller than bandwidth.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, bandwidth=10, use_cg=True, cg_max_iterations=None,
                 preconditioner_type='diagonal', use_sqrt = True):
        if preconditioner_type not in ['diagonal', None]:
            raise ValueError(f"Invalid preconditioner_type: {preconditioner_type}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, bandwidth=bandwidth,
                        use_cg=use_cg, cg_max_iterations=cg_max_iterations,
                        preconditioner_type=preconditioner_type)
        super().__init__(params, defaults)

        self.use_sqrt = use_sqrt

    @torch.no_grad
    def step(self, closure=None):
        """Performs a single optimization step.
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
                    raise RuntimeError('BandedAdam does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Second moment estimate (banded or full)
                    bandwidth = group['bandwidth']
                    if p.numel() <= bandwidth:
                        state['exp_avg_sq_full'] = torch.zeros(p.numel(), p.numel(), dtype=p.dtype, device=p.device) # Full matrix for small params
                        state['use_full_covariance'] = True
                    else:
                        state['exp_avg_banded_sq'] = [torch.zeros_like(p.data) for _ in range(bandwidth + 1)] # Banded for larger params
                        state['use_full_covariance'] = False


                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                state['step'] += 1
                use_full_covariance = state['use_full_covariance']

                # Decay weights if weight_decay is set
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # First moment update
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                m_hat = exp_avg / (1 - beta1**state['step']) # Bias correction for first moment


                if use_full_covariance: # Full covariance update for small parameters
                    exp_avg_sq_full = state['exp_avg_sq_full']
                    grad_flat = grad.flatten()
                    outer_prod = torch.outer(grad_flat, grad_flat)
                    exp_avg_sq_full.mul_(beta2).add_(1 - beta2, outer_prod)
                    V_matrix = exp_avg_sq_full / (1 - beta2**state['step']) # Bias correction for full second moment

                    def A_multiply_full(x):
                        return torch.matmul(V_matrix, x) # Dense matrix multiply
                    A_multiply = A_multiply_full
                    b = m_hat.data.flatten().clone().detach() # RHS for CG

                else: # Banded covariance update for larger parameters
                    exp_avg_banded_sq = state['exp_avg_banded_sq']
                    sq_grad = grad * grad # Main diagonal still uses squared gradient
                    exp_avg_banded_sq[0].mul_(beta2).add_(1 - beta2, sq_grad)

                    for i in range(1, group['bandwidth'] + 1):
                        # Shifted gradient product approximation for off-diagonal bands
                        grad_shifted = torch.cat([torch.zeros_like(grad[0:i]), grad[:-i]], dim=0) if i < grad.numel() else torch.zeros_like(grad) # shift by i
                        shifted_grad_product = grad * grad_shifted # Element-wise product as approximation
                        exp_avg_banded_sq[i].mul_(beta2).add_(1 - beta2, shifted_grad_product) # Update band estimate
                    # No bias correction for banded second moment in this simplified version.

                    def A_multiply_banded(x):
                        return banded_matrix_vector_multiply(exp_avg_banded_sq, x.view_as(p.data), group['bandwidth']).flatten()
                    A_multiply = A_multiply_banded
                    b = m_hat.data.flatten().clone().detach() # RHS for CG


                if group['use_cg']:
                    x0 = torch.zeros_like(b)
                    cg_max_iterations = group['cg_max_iterations'] if group['cg_max_iterations'] else p.numel()

                    # Preconditioner (Diagonal) - apply same preconditioner logic for both full and banded
                    preconditioner = None
                    if group['preconditioner_type'] == 'diagonal':
                        if use_full_covariance:
                            diag_V = torch.diag(V_matrix) # Diagonal from full covariance
                        else:
                            diag_V = exp_avg_banded_sq[0] # Diagonal from banded approx.
                        diag_band = diag_V + group['eps'] # add eps for numerical stability
                        if self.use_sqrt: M_inv_sqrt_diag = 1.0 / torch.sqrt(diag_band)
                        else: M_inv_sqrt_diag = 1 / diag_band

                        def diagonal_preconditioner(vector):
                            return M_inv_sqrt_diag * vector
                        preconditioner = diagonal_preconditioner


                    update, cg_iterations = cg(A_multiply, b, x0=x0, max_iter=cg_max_iterations,
                                                tolerance=group['eps'], preconditioner=preconditioner)
                    update_direction = update
                else: # Fallback to diagonal Adam-like update if not using CG (though CG is intended use)
                    if use_full_covariance:
                        diag_V = torch.diag(V_matrix)
                    else:
                        diag_V = exp_avg_banded_sq[0]
                    v_hat_diag = diag_V # No bias correction again, consistent with banded case.

                    if self.use_sqrt: update_direction = m_hat / (torch.sqrt(v_hat_diag) + group['eps'])
                    else: update_direction = m_hat / (v_hat_diag + group['eps'])


                p.data.add_(-group['lr'], update_direction.view_as(p.data))

        return loss