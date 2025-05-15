import math
from collections import defaultdict

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer


# Helper function to compute autocorrelation using FFT
# Autocorrelation is related to Power Spectral Density via Wiener-Khinchin theorem
# autocorr(g) = IFFT( |FFT(g)|^2 )
# We need the real part as the input g is real, and autocorr should be real.
def compute_autocorrelation_row(g_flat):
    """Computes the first row of the circulant approximation via autocorrelation."""
    if g_flat.numel() == 0:
        return torch.zeros_like(g_flat)
    fft_g = torch.fft.fft(g_flat)
    # Power Spectral Density (magnitude squared)
    psd = fft_g.abs().pow(2)
    # Inverse FFT to get autocorrelation
    autocorr = torch.fft.ifft(psd)
    # Return the real part (imaginary part should be negligible due to numerical precision)
    # The result corresponds to the first row of the empirical covariance matrix
    # projected onto the space of circulant matrices.
    return torch.real(autocorr)

class AdamCirculant(Optimizer):
    """
    Implements Adam optimizer with Circulant Approximation for the second moment.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its autocorrelation
            (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator (eigenvalues) to
            improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Initialize state (using defaultdict for convenience)
        # Store per-parameter state instead of global state dictionary
        # self.state = defaultdict(dict) # Already done by parent class Optimizer


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamCirculant does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values (m)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of autocorrelation row (c)
                    # Stores the first row of the circulant matrix C_t
                    state['exp_avg_autocorr_row'] = torch.zeros_like(p.view(-1), memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_autocorr_row = state['exp_avg_autocorr_row']
                step = state['step']

                # Increment step
                step += 1
                state['step'] = step # Store back the incremented step

                # Apply weight decay if specified
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Flatten gradient and parameter for circulant calculations
                grad_flat = grad.view(-1)
                p_flat = p.view(-1) # Needed for reshape later

                # Update biased first moment estimate (m_t)
                exp_avg_flat = exp_avg.view(-1)
                exp_avg_flat.mul_(beta1).add_(grad_flat, alpha=1 - beta1)

                # Update biased second moment estimate (c_t - autocorrelation row)
                current_autocorr_row = compute_autocorrelation_row(grad_flat)
                exp_avg_autocorr_row.mul_(beta2).add_(current_autocorr_row, alpha=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                m_hat_flat = exp_avg_flat / bias_correction1
                c_hat_flat = exp_avg_autocorr_row / bias_correction2

                # --- Preconditioning Step using Circulant Matrix C_hat ---
                # C_hat is defined by its first row c_hat_flat

                if c_hat_flat.numel() > 0:
                    # Compute eigenvalues of C_hat: FFT(c_hat_flat)
                    # Eigenvalues can be complex in theory, but for autocorrelation
                    # of real signals, they should be real and non-negative.
                    # We take torch.real for robustness.
                    eigenvalues = torch.real(torch.fft.fft(c_hat_flat))

                    # Stabilize eigenvalues: clamp below at 0 and add epsilon
                    # We compute 1 / sqrt(eigenvalues + eps) which corresponds to Lambda^{-1/2}
                    # Using rsqrt(clamp(eig, min=0) + eps) for efficiency and stability
                    inv_sqrt_eigenvalues = torch.rsqrt(torch.clamp(eigenvalues, min=0.0) + eps)

                    # Apply C_hat^{-1/2} to m_hat_flat using FFT:
                    # Result = IFFT( Lambda^{-1/2} * FFT(m_hat_flat) )
                    m_hat_fft = torch.fft.fft(m_hat_flat)
                    preconditioned_m_fft = m_hat_fft * inv_sqrt_eigenvalues
                    preconditioned_m_flat = torch.real(torch.fft.ifft(preconditioned_m_fft))

                    # Reshape the update to the original parameter shape
                    update_direction = preconditioned_m_flat.view_as(p) # Reshape using original p
                else:
                    # Handle case of 0-element tensors (e.g. tracking num batches)
                     update_direction = torch.zeros_like(p)


                # Apply the update to the parameter
                p.add_(update_direction, alpha=-lr)

                # Update state tensors (in case view created copies, though unlikely for mul_/add_)
                # state['exp_avg'] is updated in-place via exp_avg_flat view
                # state['exp_avg_autocorr_row'] is updated in-place

        return loss


class AdamFFT(optim.Optimizer):
    """
    Implements a variant of the Adam algorithm that uses an FFT-based
    circulant approximation for the second moment matrix.

    Instead of storing diagonal second moments (variances), it stores an
    exponential moving average of the power spectrum of the gradients in the
    FFT domain. This implicitly defines a circulant preconditioner.
    The preconditioning step M^{-1/2}*g is performed efficiently using FFTs:
    IFFT( FFT(g) / sqrt(EMA(|FFT(grad)|^2) ) )

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its power spectrum
            (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamFFT does not support sparse gradients')

                state = self.state[p]

                # Store original shape and flatten
                original_shape = p.shape
                # Handle 0-dim tensors (single scalars) -> view as 1-dim
                if grad.ndim == 0:
                    grad = grad.view(1)
                    p_data_flat = p.data.view(1)
                else:
                    p_data_flat = p.data.reshape(-1)
                    grad = grad.reshape(-1)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values (m)
                    state['exp_avg'] = torch.zeros_like(p_data_flat, memory_format=torch.preserve_format)
                    # Exponential moving average of gradient power spectrum (v_fft)
                    # Need size corresponding to FFT output (which is same as input for 1D FFT)
                    state['exp_avg_fft_sq'] = torch.zeros_like(p_data_flat, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_fft_sq = state['exp_avg_fft_sq']
                state['step'] += 1
                step = state['step']

                # Apply weight decay (if any)
                if weight_decay != 0:
                    grad = grad.add(p_data_flat, alpha=weight_decay)

                # Update biased first moment estimate (m_t)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute FFT of gradient and its power spectrum
                grad_fft = torch.fft.fft(grad)
                # Power spectrum |FFT(g)|^2. Must be real.
                power_spectrum = torch.real(grad_fft * torch.conj(grad_fft))
                # power_spectrum = grad_fft.abs().pow_(2) # Alternative, maybe clearer

                # Update biased second raw moment estimate in FFT domain (v_fft_t)
                exp_avg_fft_sq.mul_(beta2).add_(power_spectrum, alpha=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_fft_hat = exp_avg_fft_sq / bias_correction2

                # === Preconditioning Step ===
                # Transform m_hat to FFT domain
                m_hat_fft = torch.fft.fft(m_hat)

                # Denominator: sqrt(v_fft_hat) + eps
                # Ensure v_fft_hat is non-negative before sqrt (should be, but for safety)
                v_fft_hat_sqrt = torch.sqrt(torch.clamp(v_fft_hat, min=0.0)) + eps

                # Apply preconditioner in FFT domain
                preconditioned_m_fft = m_hat_fft / v_fft_hat_sqrt

                # Transform back to parameter domain
                update_direction_flat = torch.fft.ifft(preconditioned_m_fft)

                # Ensure update is real (discard small imaginary parts due to numerical errors)
                update_direction_flat = torch.real(update_direction_flat)
                # ============================

                # Reshape update back to original parameter shape
                update_direction = update_direction_flat.reshape(original_shape)

                # Apply update to parameters
                p.add_(update_direction, alpha=-lr)

        return loss

# Example Usage (similar to how you use torch.optim.Adam)
# model = YourModel()
# optimizer = AdamFFT(model.parameters(), lr=0.001, betas=(0.9, 0.99))
#
# ... training loop ...
# optimizer.zero_grad()
# loss = compute_loss(model(inputs), targets)
# loss.backward()
# optimizer.step()


import torch

def cg_solve(A_mvp, b, x0=None, tol=1e-5, max_iter=None, eps_reg=1e-8):
    """
    Solves the linear system A*x = b using the Conjugate Gradient method.

    Args:
        A_mvp (callable): A function that takes a vector x and returns A@x.
                          Handles the implicit matrix A. IMPORTANT: A MUST BE SPD.
                          The S_hat + eps*I matrix should be SPD.
        b (torch.Tensor): The right-hand side vector.
        x0 (torch.Tensor, optional): Initial guess for the solution. Defaults to zeros.
        tol (float, optional): Tolerance for convergence (relative residual norm).
        max_iter (int, optional): Maximum number of iterations. Defaults to len(b).
        eps_reg (float, optional): Small value added to denominators for numerical stability.

    Returns:
        torch.Tensor: The solution vector x.
    """
    n = b.numel()
    if max_iter is None:
        max_iter = n * 2 # Often converges faster, but set a safe default

    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A_mvp(x)

    p = r.clone()
    rs_old = torch.dot(r, r)

    if rs_old.sqrt() < tol: # Already converged
        return x

    for i in range(max_iter):
        Ap = A_mvp(p)
        alpha = rs_old / (torch.dot(p, Ap) + eps_reg) # Add eps for stability

        x += alpha * p
        r -= alpha * Ap

        rs_new = torch.dot(r, r)
        if rs_new.sqrt() < tol:
            # print(f"CG converged in {i+1} iterations.")
            break

        beta = rs_new / (rs_old + eps_reg) # Add eps for stability
        p = r + beta * p
        rs_old = rs_new
    # else:
        # print(f"CG reached max iterations ({max_iter}).")


    return x

import torch.fft

def _diagonal_mvp(diag_vec, x):
    """Matrix-vector product for a diagonal matrix."""
    return diag_vec * x

def _circulant_mvp(c_first_col, x):
    """
    Matrix-vector product for a Circulant matrix C(c) using FFT.
    y = C(c) @ x = ifft(fft(c) * fft(x))
    Assumes PyTorch's default backward normalization for fft/ifft.
    """
    n = c_first_col.numel()
    if n == 0: return torch.zeros_like(x)
    # Ensure complex fft needed if either is complex, otherwise real->real is fine
    is_complex = c_first_col.is_complex() or x.is_complex()

    fft_c = torch.fft.fft(c_first_col)
    fft_x = torch.fft.fft(x)

    # Element-wise product in frequency domain
    fft_y = fft_c * fft_x

    # Inverse FFT to get result in time domain
    y = torch.fft.ifft(fft_y)

    # Return real part if inputs were real
    return y.real if not is_complex else y

def _toeplitz_mvp(t_first_col, t_first_row, x):
    """
    Matrix-vector product for a Toeplitz matrix T(c, r) using FFT via Circulant embedding.
    y = T @ x
    """
    n = t_first_col.numel()
    if n == 0: return torch.zeros_like(x)
    if n == 1: return t_first_col * x # Handle scalar case

    # Ensure t_first_col[0] == t_first_row[0]
    # (Should be handled by update logic, but good practice)
    # t_first_row[0] = t_first_col[0] # Or average them if inconsistent

    # Construct the first column of the embedding Circulant matrix (size 2N-1 or 2N)
    # Using size 2N is often simpler for FFT libraries
    fft_len = 2 * n

    # Circulant column 'c_emb': [c[0], c[1], ..., c[N-1], 0, r[N-1], ..., r[1]]
    # Pad t_first_col: [c[0], ..., c[N-1], 0, ..., 0] length fft_len
    c_pad = torch.nn.functional.pad(t_first_col, (0, fft_len - n))
    # Pad flipped t_first_row[1:]: [0, r[N-1], ..., r[1], 0, ..., 0] length fft_len
    # Need to handle n=1 case for r[1:]
    r_rev_pad = torch.nn.functional.pad(torch.flip(t_first_row[1:], dims=[0]), (1, fft_len - n)) # Add 1 zero at start

    c_embedded = c_pad + r_rev_pad
    c_embedded[0] = t_first_col[0] # Ensure correct diagonal element

    # Pad x to fft_len
    x_padded = torch.nn.functional.pad(x, (0, fft_len - n))

    # Perform circulant MVP using FFT
    y_padded = _circulant_mvp(c_embedded, x_padded)

    # Extract the first N elements
    y = y_padded[:n]
    return y

import math
from torch.optim.optimizer import Optimizer

# Re-use projection logic, ensuring PyTorch operations
def _update_diagonal_params(grad_flat, state, beta2, device):
    if 'exp_avg_sq_diag' not in state:
        state['exp_avg_sq_diag'] = torch.zeros_like(grad_flat) # Store as flat vector
    exp_avg_sq_diag = state['exp_avg_sq_diag']
    # Update using g^2
    update_diag = grad_flat * grad_flat
    exp_avg_sq_diag.mul_(beta2).add_(update_diag, alpha=1 - beta2)
    return exp_avg_sq_diag

def _update_circulant_params(grad_flat, state, beta2, device):
    n = grad_flat.numel()
    if 'exp_avg_sq_circ_c' not in state:
        state['exp_avg_sq_circ_c'] = torch.zeros(n, dtype=grad_flat.dtype, device=device)
    exp_avg_sq_c = state['exp_avg_sq_circ_c']

    # Projection using FFT Autocorrelation: c_k = mean(g_i * g_{(i+k)%n})
    # Autocorr(g) = IFFT( |FFT(g)|^2 ) / n (for mean)
    grad_fft = torch.fft.fft(grad_flat)
    psd = grad_fft * torch.conj(grad_fft) # Power spectral density
    # Use PyTorch's default norm="backward" for FFT/IFFT pair: IFFT(FFT(x)) = x
    # The definition c_k = mean(...) implies division by n.
    autocorr = torch.fft.ifft(psd).real # ifft returns complex, take real part
    update_c = autocorr / n

    exp_avg_sq_c.mul_(beta2).add_(update_c, alpha=1 - beta2)
    return exp_avg_sq_c

def _update_toeplitz_params(grad_flat, state, beta2, device):
    n = grad_flat.numel()
    if 'exp_avg_sq_toep_c' not in state:
        state['exp_avg_sq_toep_c'] = torch.zeros(n, dtype=grad_flat.dtype, device=device)
        state['exp_avg_sq_toep_r'] = torch.zeros(n, dtype=grad_flat.dtype, device=device)

    exp_avg_sq_tc = state['exp_avg_sq_toep_c']
    exp_avg_sq_tr = state['exp_avg_sq_toep_r']

    # Projection using FFT for non-circular autocorrelation
    # Average value along each diagonal k = j - i
    N = n
    if N == 0: return exp_avg_sq_tc, exp_avg_sq_tr # Handle empty grad
    if N == 1: # Handle scalar case explicitly
        update_val = grad_flat * grad_flat
        exp_avg_sq_tc.mul_(beta2).add_(update_val, alpha=1 - beta2)
        exp_avg_sq_tr.copy_(exp_avg_sq_tc) # Ensure consistency
        return exp_avg_sq_tc, exp_avg_sq_tr

    fft_len = 2 * N -1 # Minimum length for linear convolution via FFT

    # Compute autocorr using FFT: conv(g, flip(conj(g)))
    fft_g = torch.fft.fft(grad_flat, n=fft_len)
    # Need g_rev_conj = flip(conj(g)) padded appropriately
    g_rev_conj = torch.conj(torch.flip(grad_flat, dims=[0]))
    fft_g_rev_conj = torch.fft.fft(g_rev_conj, n=fft_len)

    conv_fft = fft_g * fft_g_rev_conj
    autocorr_full = torch.fft.ifft(conv_fft).real # Result length fft_len

    # Extract diagonal sums:
    # Sum for diag k=j-i: index N-1+k in autocorr_full
    # First column c (k=0..N-1): Indices N-1 to 2N-2
    update_tc_sums = autocorr_full[N-1:]
    # First row r (k=0..-(N-1) => -k=0..N-1): Indices N-1 down to 0
    update_tr_sums = torch.flip(autocorr_full[:N], dims=[0]) # autocorr_full[N-1], ..., autocorr_full[0]

    # Get counts for averaging each diagonal
    counts = torch.arange(N, 0, -1, device=device, dtype=grad_flat.dtype) # N, N-1, ..., 1

    update_tc = update_tc_sums / counts
    update_tr = update_tr_sums / counts
    # Ensure c[0] == r[0] (they are computed from the same diagonal sum / count)
    update_tc[0] = update_tc_sums[0] / N
    update_tr[0] = update_tc[0]

    # EMA Update
    exp_avg_sq_tc.mul_(beta2).add_(update_tc, alpha=1 - beta2)
    exp_avg_sq_tr.mul_(beta2).add_(update_tr, alpha=1 - beta2)
    # Ensure consistency for the first element after EMA
    exp_avg_sq_tc[0] = (exp_avg_sq_tc[0] + exp_avg_sq_tr[0]) / 2.0
    exp_avg_sq_tr[0] = exp_avg_sq_tc[0]

    return exp_avg_sq_tc, exp_avg_sq_tr


class StructuredAdam(Optimizer):
    r"""Implements Adam algorithm with combined structured covariance matrices
       (Diagonal, Circulant, Toeplitz) using PyTorch native operations and CG solver.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator (and diagonal of matrix
             for CG solve) to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        use_diagonal (bool, optional): Whether to include the diagonal component
            of the preconditioner (default: True)
        use_circulant (bool, optional): Whether to include the circulant component
            (default: False)
        use_toeplitz (bool, optional): Whether to include the toeplitz component
            (default: False)
        cg_tolerance (float, optional): Tolerance for the Conjugate Gradient solver
            (default: 1e-5)
        cg_max_iter (int, optional): Maximum iterations for CG solver.
            Defaults to 2 * param_dim. (default: None)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) - Note: AMSGrad logic needs careful adaptation for
            structured matrices, using max of matrix parameters is non-trivial.
            *Currently not fully implemented for structured part.*
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, use_diagonal=True, use_circulant=False,
                 use_toeplitz=False, cg_tolerance=1e-5, cg_max_iter=None,
                 amsgrad=False): # Added amsgrad placeholder

        if not 0.0 <= lr: raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps: raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0: raise ValueError("Invalid beta1: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0: raise ValueError("Invalid beta2: {}".format(betas[1]))
        if not 0.0 <= weight_decay: raise ValueError("Invalid weight_decay: {}".format(weight_decay))
        if not (use_diagonal or use_circulant or use_toeplitz):
             raise ValueError("At least one matrix structure must be enabled.")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        use_diagonal=use_diagonal, use_circulant=use_circulant,
                        use_toeplitz=use_toeplitz, cg_tolerance=cg_tolerance,
                        cg_max_iter=cg_max_iter, amsgrad=amsgrad) # Added amsgrad
        super(StructuredAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(StructuredAdam, self).__setstate__(state)
        # Set amsgrad default for loaded state, if needed
        for group in self.param_groups:
            group.setdefault('amsgrad', False)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            use_diagonal = group['use_diagonal']
            use_circulant = group['use_circulant']
            use_toeplitz = group['use_toeplitz']
            cg_tol = group['cg_tolerance']
            cg_max_iter = group['cg_max_iter']
            amsgrad = group['amsgrad'] # Get amsgrad flag

            # Check if amsgrad is used with structured matrices - warn if not fully supported
            if amsgrad and (use_circulant or use_toeplitz):
                # warnings.warn("AMSGrad implementation is currently basic for structured matrices.")
                pass # Proceed with caution

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('StructuredAdam does not support sparse gradients.')

                state = self.state[p]
                param_dim = p.numel()
                param_shape = p.shape
                grad_flat = grad.flatten() # Work with flattened gradient/params

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(grad_flat) # Store flat
                    if use_diagonal:
                        state['exp_avg_sq_diag'] = torch.zeros_like(grad_flat)
                        if amsgrad:
                           state['max_exp_avg_sq_diag'] = torch.zeros_like(grad_flat)
                    if use_circulant:
                        state['exp_avg_sq_circ_c'] = torch.zeros(param_dim, dtype=grad_flat.dtype, device=grad.device)
                        # Add AMSGrad state if needed (tricky: max of vectors?)
                    if use_toeplitz:
                        state['exp_avg_sq_toep_c'] = torch.zeros(param_dim, dtype=grad_flat.dtype, device=grad.device)
                        state['exp_avg_sq_toep_r'] = torch.zeros(param_dim, dtype=grad_flat.dtype, device=grad.device)
                        # Add AMSGrad state if needed

                exp_avg = state['exp_avg']
                state['step'] += 1
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']

                # Apply weight decay (L2 penalty)
                if weight_decay != 0:
                     grad_flat = grad_flat.add(p.flatten(), alpha=weight_decay) # Apply to flat grad

                # Decay the first moment running average coefficient
                exp_avg.mul_(beta1).add_(grad_flat, alpha=1 - beta1)
                m_hat = exp_avg / bias_correction1 # Bias corrected m

                # --- Update second moment estimates for enabled structures ---
                current_diag_hat = None
                current_c_hat = None
                current_tc_hat = None
                current_tr_hat = None

                if use_diagonal:
                    v_diag = _update_diagonal_params(grad_flat, state, beta2, grad.device)
                    if amsgrad:
                        # Maintain the maximum of all 2nd moment running avg till now
                        max_v_diag = state['max_exp_avg_sq_diag']
                        torch.maximum(max_v_diag, v_diag, out=max_v_diag)
                        # Use the max. for preconditioning
                        current_diag_hat = max_v_diag / bias_correction2 # Bias correct the max
                    else:
                         current_diag_hat = v_diag / bias_correction2

                if use_circulant:
                    v_circ_c = _update_circulant_params(grad_flat, state, beta2, grad.device)
                    # TODO: Add AMSGrad logic for circulant if desired (e.g., element-wise max of 'c'?)
                    current_c_hat = v_circ_c / bias_correction2

                if use_toeplitz:
                    v_toep_c, v_toep_r = _update_toeplitz_params(grad_flat, state, beta2, grad.device)
                    # TODO: Add AMSGrad logic for toeplitz if desired
                    current_tc_hat = v_toep_c / bias_correction2
                    current_tr_hat = v_toep_r / bias_correction2
                    # Ensure consistency again after bias correction? Should be consistent before.
                    current_tr_hat[0] = current_tc_hat[0]

                # --- Define the MVP function for CG ---
                def combined_mvp(x):
                    y = torch.zeros_like(x)
                    if use_diagonal and current_diag_hat is not None:
                        # Clamp negative values resulting from floating point errors?
                        # Add small epsilon before sqrt in standard Adam, here add to matrix diag
                        # y += _diagonal_mvp(current_diag_hat.clamp(min=0), x)
                        y += _diagonal_mvp(current_diag_hat, x) # Add eps later
                    if use_circulant and current_c_hat is not None:
                        y += _circulant_mvp(current_c_hat, x)
                    if use_toeplitz and current_tc_hat is not None and current_tr_hat is not None:
                        y += _toeplitz_mvp(current_tc_hat, current_tr_hat, x)

                    # Add regularization term eps * I
                    y.add_(x, alpha=eps)
                    return y

                # --- Solve the system using CG ---
                # S_hat should be positive semi-definite. Adding eps*I makes it positive definite.
                if param_dim > 0:
                    update_direction = cg_solve(combined_mvp, m_hat,
                                                tol=cg_tol,
                                                max_iter=cg_max_iter if cg_max_iter is not None else 2 * param_dim,
                                                eps_reg=eps*eps) # Use small eps for CG stability
                else:
                    update_direction = torch.tensor(0.0, device=p.device)


                # --- Apply update ---
                update_reshaped = update_direction.reshape(param_shape)
                p.add_(update_reshaped, alpha=-lr)

        return loss