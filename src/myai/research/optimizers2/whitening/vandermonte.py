import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import math
import warnings

# Helper function to create Vandermonde matrix
def vandermonde(points, degree):
    """
    Creates a Vandermonde matrix.
    V[i, j] = points[i] ** j
    Args:
        points (Tensor): 1D Tensor of points (size N).
        degree (int): The maximum degree (M-1), resulting in M columns (0 to M-1).
    Returns:
        Tensor: Vandermonde matrix of shape (N, M).
    """
    N = points.size(0)
    M = degree + 1
    if M <= 0:
        return torch.empty(N, 0, dtype=points.dtype, device=points.device)
    # Handle degree 0 separately to avoid 0**0 issues if points contains 0
    V = torch.empty(N, M, dtype=points.dtype, device=points.device)
    V[:, 0] = 1.0
    for j in range(1, M):
        V[:, j] = points * V[:, j - 1] # More stable than points ** j
    return V

class AdamV(Optimizer):
    """
    Implements a variant of the Adam algorithm using a Vandermonde-projection
    for the second moment estimate.

    The second moment information is projected onto a subspace defined by the
    columns of a Vandermonde matrix B (size d x k).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-3).
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its projected square (default: (0.9, 0.999)).
        eps (float, optional): term added to the denominator in the projected space
            to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
        k (int): Dimension of the Vandermonde subspace (rank). Must be > 0.
                 Effective k will be min(k, param_dim).
        amsgrad (boolean, optional): whether to use the AMSGrad variant (default: False).
        vandermonde_points (str, optional): Method to generate points for Vandermonde rows.
                 Currently only 'linspace' supported. (default: 'linspace').
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k=10, amsgrad=False, vandermonde_points='linspace'):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Invalid k value: {} (must be positive integer)".format(k))
        if vandermonde_points != 'linspace':
             raise ValueError("Only 'linspace' is supported for vandermonde_points currently.")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, k=k,
                        amsgrad=amsgrad, vandermonde_points=vandermonde_points)
        super(AdamV, self).__init__(params, defaults)

        # Cache for basis matrices keyed by (dim, k, device)
        self._basis_cache = {}

    def _get_vandermonde_basis(self, dim, k, device):
        """
        Retrieves or computes the Vandermonde basis matrix B (d x k_eff) and B_T.
        k_eff is min(k, dim).
        """
        if dim <= 0: # Should not happen for valid params
             return None, None, 0

        # Effective k cannot be larger than the parameter dimension
        k_eff = min(k, dim)
        if k_eff == 0: # If k=0 or dim=0
            return None, None, 0

        cache_key = (dim, k_eff, device)
        if cache_key in self._basis_cache:
            return self._basis_cache[cache_key] + (k_eff,)

        # --- Compute Basis ---
        # Generate 'dim' points for the rows of the Vandermonde matrix
        # Using linspace is a simple, fixed choice.
        points = torch.linspace(-1, 1, dim, device=device, dtype=torch.float32) # Use float32 for stability

        # Create Vandermonde matrix B (dim x k_eff). Degree is k_eff - 1.
        B = vandermonde(points, k_eff - 1)

        # Optional: Normalize columns of B for better conditioning
        # This might deviate from a pure Vandermonde structure but improves stability
        B_norm = torch.norm(B, p=2, dim=0, keepdim=True)
        # Add small epsilon to avoid division by zero for potential zero columns (e.g., k=1)
        B = B / (B_norm + 1e-8)

        B_T = B.t().detach() # Transpose and detach
        B = B.detach()      # Detach original too
        # --- End Compute Basis ---

        self._basis_cache[cache_key] = (B, B_T)
        return B, B_T, k_eff

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
            k = group['k']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamV does not support sparse gradients')

                state = self.state[p]
                param_dim = p.numel()
                if param_dim == 0: continue # Skip empty parameters

                # Get Vandermonde basis B (d x k_eff), B_T (k_eff x d)
                B, B_T, k_eff = self._get_vandermonde_basis(param_dim, k, p.device)

                # Ensure k_eff > 0
                if k_eff <= 0:
                    warnings.warn(f"Skipping parameter with dim {param_dim} <= k_eff {k_eff}")
                    continue

                # Ensure B, B_T are on the correct device and dtype
                # (Should be handled by _get_vandermonde_basis cache key, but double check)
                if B.device != p.device or B.dtype != p.dtype:
                     B = B.to(p.device, p.dtype)
                     B_T = B_T.to(p.device, p.dtype)


                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    # First moment estimate (vector, size d)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Projected second moment estimate (vector, size k_eff)
                    state['exp_avg_sq_proj'] = torch.zeros(k_eff, dtype=p.dtype, device=p.device)
                    if amsgrad:
                        # Maintains max of projected exp. avg. squares (size k_eff)
                        state['max_exp_avg_sq_proj'] = torch.zeros(k_eff, dtype=p.dtype, device=p.device)
                # Ensure state tensors for projected values match k_eff if k changed or first step
                elif state['exp_avg_sq_proj'].shape[0] != k_eff:
                     state['exp_avg_sq_proj'] = torch.zeros(k_eff, dtype=p.dtype, device=p.device)
                     if amsgrad:
                         state['max_exp_avg_sq_proj'] = torch.zeros(k_eff, dtype=p.dtype, device=p.device)


                exp_avg = state['exp_avg']
                exp_avg_sq_proj = state['exp_avg_sq_proj']
                if amsgrad:
                    max_exp_avg_sq_proj = state['max_exp_avg_sq_proj']

                state['step'] += 1
                step = state['step']

                # Perform weight decay (AdamW style)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay) # In-place easier than adding to grad sometimes

                # Decay the first moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Project gradient onto basis B
                # Reshape grad to be a flat vector [d, 1] for projection
                grad_flat = grad.reshape(-1, 1).to(B_T.dtype) # Ensure dtype match
                g_proj = torch.mm(B_T, grad_flat).squeeze() # Result is size k_eff

                # Update projected second moment estimate s
                exp_avg_sq_proj.mul_(beta2).addcmul_(g_proj, g_proj, value=1 - beta2)

                # --- Calculate denominator for the update in projected space ---
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq_proj, exp_avg_sq_proj, out=max_exp_avg_sq_proj)
                    # Use the max. for update
                    denom_proj_unbiased = max_exp_avg_sq_proj.sqrt() # Already includes bias correction implicitly over steps
                    # Apply bias correction term explicitly for stability, matching Adam's approach
                    bias_correction2 = 1 - beta2 ** step
                    denom_proj = denom_proj_unbiased / math.sqrt(bias_correction2) + eps

                else:
                    bias_correction2 = 1 - beta2 ** step
                    # Add eps *after* sqrt and bias correction for consistency with PyTorch Adam
                    denom_proj = (exp_avg_sq_proj / bias_correction2).sqrt().add_(eps)
                    #denom_proj = exp_avg_sq_proj.sqrt() / math.sqrt(bias_correction2) + eps # Equivalent

                # --- Calculate numerator for the update in projected space ---
                bias_correction1 = 1 - beta1 ** step
                m_hat = exp_avg / bias_correction1

                # Project bias-corrected first moment m_hat
                # Reshape m_hat to be a flat vector [d, 1] for projection
                m_hat_flat = m_hat.reshape(-1, 1).to(B_T.dtype) # Ensure dtype match
                m_hat_proj = torch.mm(B_T, m_hat_flat).squeeze() # Result is size k_eff

                # --- Compute update step ---
                # Scale in projected space
                step_proj = m_hat_proj / denom_proj # size k_eff

                # Project step back to full parameter space
                # step_proj is size k_eff, needs to be [k_eff, 1] for mm
                update_flat = torch.mm(B, step_proj.unsqueeze(1)) # Result is [d, 1]

                # Reshape update to match parameter shape
                update = update_flat.reshape(p.shape).to(p.dtype) # Ensure dtype match

                # Apply update: p = p - lr * update
                # Note: Weight decay was applied *before* gradient processing here
                p.add_(update, alpha=-lr)

        return loss