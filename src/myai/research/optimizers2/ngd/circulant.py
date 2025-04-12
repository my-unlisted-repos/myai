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