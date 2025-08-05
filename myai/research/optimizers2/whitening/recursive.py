import math
import warnings  # For potential padding warnings

import torch
import torch.optim as optim
from torch.nn.utils import parameters_to_vector


def _inv_sqrt_2x2_batch(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse square root of a possibly batched 2x2 matrix."""
    a = A[..., 0, 0]
    b = A[..., 0, 1]
    c = A[..., 1, 0]
    d = A[..., 1, 1]

    det = (a * d).sub_(b * c)
    trace = a + d

    # add smallest eigenvalue to diagonal to force PD
    term1 = trace/2
    term2 = (trace.pow(2).div_(4).sub_(det)).clamp_(min=eps).sqrt_()
    y1 = term1 + term2
    y2 = term1 - term2
    smallest_eigval = torch.minimum(y1, y2).neg_().clamp_(min=0) + eps
    a = a+smallest_eigval
    d = d+smallest_eigval

    # recalculate det and trace witg new a and b
    det = (a * d).sub_(b * c)
    trace = a + d


    s = (det.clamp(min=eps)).sqrt_()

    tau_squared = trace + 2 * s
    tau = (tau_squared.clamp(min=eps)).sqrt_()

    denom = s * tau

    coeff = (denom.clamp(min=eps)).reciprocal_().unsqueeze(-1).unsqueeze(-1)

    row1 = torch.stack([d + s, -b], dim=-1)
    row2 = torch.stack([-c, a + s], dim=-1)
    M = torch.stack([row1, row2], dim=-2)

    return coeff * M

class RecursivePreconditionedAdam(optim.Optimizer):
    """
    Implements Adam with recursive 2x2 block preconditioning replacing the
    second moment estimate.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate (default: 1e-3)
        beta1 (float, optional): Coefficient for momentum (default: 0.9)
        weight_decay (float, optional): Weight decay (AdamW style) (default: 0)
        precond_beta (float, optional): Decay rate for preconditioner matrices (default: 0.99)
        precond_eps (float, optional): Stability epsilon for inv sqrt (default: 1e-6)
        update_interval (int, optional): Update preconditioner every N steps (default: 1)
        max_levels (int, optional): Maximum number of recursive levels. If None,
            levels are chosen to potentially reach a single block at the top,
            requiring power-of-2 padding. If set, padding is minimized to
            the granularity required by max_levels. (default: None)
    """
    def __init__(self, params, lr=1e-3, beta1=0.9,
                 weight_decay=0, precond_beta=0.99, precond_eps=1e-6,
                 update_interval=1, max_levels=None):

        defaults = dict(lr=lr, beta1=beta1, weight_decay=weight_decay,
                        precond_beta=precond_beta, precond_eps=precond_eps,
                        update_interval=update_interval, max_levels=max_levels)
        super().__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise ValueError("RecursivePreconditionedAdam doesn't support multiple parameter groups yet.")

        self._params: list[torch.Tensor] = self.param_groups[0]['params']
        self._numel_total: int = sum(p.numel() for p in self._params if p.requires_grad)

        # --- Efficient Padding Calculation ---
        self._padded_numel = self._numel_total
        self._num_levels = 0

        if self._numel_total >= 2:
            # Max possible levels L such that 2^L elements are needed at the top level (1 block)
            # N_padded >= 2^L. Or, max L such that N_padded/2^L >= 1.
            max_possible_levels = int(math.floor(math.log2(self._numel_total)))
            # However, level k=L-1 needs N divisible by 2^L.

            # Determine the target number of levels
            target_levels = max_possible_levels
            if max_levels is not None and max_levels >= 0:
                target_levels = min(max_possible_levels, max_levels)
                # If max_levels is 0, we use 0 levels.
                if max_levels == 0: target_levels = 0


            if target_levels > 0:
                # Granularity required for L levels (P_0 .. P_{L-1})
                # Level L-1 operates on blocks representing 2^L original elements.
                # Thus, the total size must be divisible by 2^L.
                granularity = 2**target_levels

                # Calculate padded size: round up numel_total to nearest multiple of granularity
                self._padded_numel = math.ceil(self._numel_total / granularity) * granularity
                self._num_levels = target_levels
            else:
                # 0 levels requested or possible. Ensure padded_numel is at least even if > 0.
                if self._numel_total > 0 and self._numel_total % 2 != 0:
                    self._padded_numel = self._numel_total + 1
                else:
                    self._padded_numel = self._numel_total # Already even or zero
                self._num_levels = 0 # Explicitly set to 0


        # Ensure padded is at least original and even if non-zero
        self._padded_numel = max(self._numel_total, self._padded_numel)
        if self._padded_numel > 0 and self._padded_numel % 2 != 0:
            # This case should technically be covered by granularity logic if L>=1
            # but as a safeguard, make it even.
            self._padded_numel += 1


        # print(f"RecursivePreconditionedAdam initialized:")
        # print(f"  Total params: {self._numel_total}")
        # if self._padded_numel != self._numel_total:
        #      print(f"  Padded params: {self._padded_numel} (padding {self._padded_numel - self._numel_total} elements)")
        # else:
        #      print(f"  Padded params: {self._padded_numel} (no padding needed)")
        # print(f"  Using Levels L = {self._num_levels} (indices 0 to {self._num_levels-1})")
        if self._numel_total > 0 and self._num_levels == 0:
            warnings.warn("Using 0 levels of preconditioning. Check max_levels or parameter count.")


        self._state_initialized = False

    def _init_state(self, p_vec_example: torch.Tensor):
        """Initialize optimizer state."""
        state = self.state['global_state'] = {}
        state['step'] = 0
        state['exp_avg'] = torch.zeros(self._numel_total, device=p_vec_example.device, dtype=p_vec_example.dtype)
        state['preconditioners'] = []

        current_padded_dim = self._padded_numel
        for k in range(self._num_levels):
            # Num blocks for P_k = (total elements) / (elements per block represented by P_k)
            # P_k combines pairs from level k+1 -> represents 2 * 2^(k+1) = 2^(k+1) elements? No.
            # P_k combines pairs of elements -> handles 2 elements per block
            # The number of blocks decreases by factor of 2 each level.
            # Level 0 has N_pad / 2 blocks. Level k has N_pad / 2^(k+1) blocks.
            num_blocks_k = current_padded_dim // 2
            if num_blocks_k < 1:
                # This shouldn't happen with correct padding logic, but safeguard.
                print(f"Warning: Calculated 0 blocks at level {k}. Adjusting num_levels.")
                self._num_levels = k
                break

            init_val = 1.0
            P_k = torch.eye(2, device=p_vec_example.device, dtype=p_vec_example.dtype).expand(num_blocks_k, 2, 2) * init_val
            state['preconditioners'].append(P_k.clone())
            current_padded_dim //= 2 # The effective dimension halves for the next level up

        self._state_initialized = True
        # print(f"Optimizer state initialized on device: {p_vec_example.device}")


    # --- _get_flat_grad (Unchanged) ---
    def _get_flat_grad(self) -> torch.Tensor | None:
        """Concatenates gradients into a single flat vector, filling zeros for missing grads."""
        grads = []
        any_valid_grad = False
        device = None
        dtype = None

        # Determine device and dtype from the first available parameter/gradient
        for p in self._params:
             # Find first parameter that requires grad to determine device/dtype context
            if p.requires_grad:
                device = p.device
                dtype = p.dtype
                if p.grad is not None:
                    # Prefer gradient's dtype if available, might be float32 even if model is float16
                    dtype = p.grad.dtype
                break # Found context, exit loop
        if device is None: # No parameters require grad
            return None

        for p in self._params:
            if p.requires_grad:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('RecursivePreconditionedAdam does not support sparse gradients')
                    grads.append(p.grad.detach().reshape(-1))
                    any_valid_grad = True
                else:
                     # If requires_grad is True but grad is None, fill with zeros
                    grads.append(torch.zeros(p.numel(), device=device, dtype=dtype))
            # else: parameter does not require grad, skip

        if not any_valid_grad and self._numel_total > 0:
             # Pass - grads list will contain all zeros if needed
            pass

        if not grads: # No parameters required grad
            return None

        flat_grad = torch.cat(grads)

        if flat_grad.numel() != self._numel_total:
            # Check consistency after potential zero-filling
            raise RuntimeError(f"Internal error: Total gradient elements ({flat_grad.numel()}) mismatch expected ({self._numel_total})")
        return flat_grad

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # --- 1. Get Gradient ---
        flat_grad = self._get_flat_grad()
        if flat_grad is None or self._numel_total == 0:
            return loss

        # --- 2. Initialize State (if first step) ---
        if not self._state_initialized:
            self._init_state(flat_grad)

        state = self.state['global_state']
        state['step'] += 1
        step = state['step']

        # --- 3. Load Hyperparameters ---
        group = self.param_groups[0]
        lr = group['lr']
        beta1 = group['beta1']
        weight_decay = group['weight_decay']
        precond_beta = group['precond_beta']
        precond_eps = group['precond_eps']
        update_interval = group['update_interval']

        # --- 4. Update Momentum ---
        exp_avg = state['exp_avg']
        exp_avg.mul_(beta1).add_(flat_grad, alpha=1.0 - beta1)

        # --- 5. Pad Gradient for Preconditioner Update (only if needed) ---
        if step % update_interval == 0 and self._num_levels > 0:
            if self._padded_numel > self._numel_total:
                padded_grad = torch.zeros(self._padded_numel, device=flat_grad.device, dtype=flat_grad.dtype)
                padded_grad[:self._numel_total] = flat_grad
            else:
                padded_grad = flat_grad # No padding necessary

            # --- Update Preconditioners (Bottom-Up using padded raw gradient) ---
            for k in range(self._num_levels):
                if k >= len(state['preconditioners']): break # Safety check

                P_k = state['preconditioners'][k]
                num_blocks_k = P_k.shape[0]
                expected_elements_k = num_blocks_k * 2

                # Determine the segment of padded_grad relevant for level k
                # Level k uses the first N_pad / 2^k elements
                segment_len = self._padded_numel // (2**k)
                if segment_len < expected_elements_k:
                    print(f"Warning: P update grad segment size ({segment_len}) insufficient for level {k} ({expected_elements_k}). Skipping.")
                    # This indicates an issue with padding/level calculation logic
                    break

                # Reshape the *start* of the padded gradient vector appropriately
                g_segment = padded_grad[:expected_elements_k] # P_k works on pairs, so takes 2*num_blocks elements
                g_blocks = g_segment.view(num_blocks_k, 2)

                # Compute batch of outer products
                outer_prods = torch.einsum('bi,bj->bij', g_blocks, g_blocks)

                # Update P_k: EMA
                P_k.lerp_(outer_prods, weight=1.0 - precond_beta)
                #P_k.mul_(precond_beta).add_(outer_prods, alpha=1.0 - precond_beta)

        # --- 6. Prepare Momentum for Preconditioning ---
        bias_correction1 = 1.0 - beta1 ** step
        m_corr = exp_avg / bias_correction1

        # Pad bias-corrected momentum (only if needed)
        if self._padded_numel > self._numel_total:
            padded_m_corr = torch.zeros(self._padded_numel, device=m_corr.device, dtype=m_corr.dtype)
            padded_m_corr[:self._numel_total] = m_corr
        else:
            padded_m_corr = m_corr # No padding necessary

        # --- 7. Apply Preconditioning to Momentum (Top-Down) ---
        m_prec = padded_m_corr.clone()
        if self._num_levels > 0:
            for k in range(self._num_levels - 1, -1, -1): # Loop L-1 down to 0
                if k >= len(state['preconditioners']): continue # Safety

                P_k = state['preconditioners'][k]
                num_blocks_k = P_k.shape[0]
                current_segment_dim = 2 * num_blocks_k # P_k operates on this many elements

                if m_prec.numel() < current_segment_dim:
                    print(f"Warning: Momentum vec dim ({m_prec.numel()}) too small for level {k} ({current_segment_dim}). Skipping P apply.")
                    continue

                # Apply P_k to the start of the current m_prec vector
                m_segment_to_precondition = m_prec[:current_segment_dim]
                m_blocks = m_segment_to_precondition.view(num_blocks_k, 2)

                try:
                    inv_sqrt_Pk = _inv_sqrt_2x2_batch(P_k, precond_eps)
                except ValueError as e:
                    print(f"Error in inv_sqrt at level {k}: {e}. Skipping P apply.")
                    continue

                # Apply preconditioning
                m_preconditioned_blocks = torch.einsum('bij,bj->bi', inv_sqrt_Pk, m_blocks)

                # Update the relevant part of m_prec
                m_prec[:current_segment_dim] = m_preconditioned_blocks.flatten()

        # --- 8. Unpad Preconditioned Momentum ---
        final_update_direction = m_prec[:self._numel_total]

        # --- 9. Update Parameters ---
        offset = 0
        for p in self._params:
            if p.requires_grad:
                numel = p.numel()
                if numel == 0: continue

                delta = final_update_direction[offset : offset + numel].view_as(p)

                # Apply weight decay (AdamW style)
                if weight_decay != 0:
                    p.add_(p, alpha= -lr * weight_decay)

                # Apply main update step
                p.add_(delta, alpha=-lr)

                offset += numel

        if offset != self._numel_total:
            print(f"Warning: Offset ({offset}) != total elements ({self._numel_total}) after update.")

        return loss