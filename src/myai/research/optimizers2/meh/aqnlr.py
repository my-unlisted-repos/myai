import torch
from torch.optim.optimizer import Optimizer
import math


class AQNLR(Optimizer):
    """Adaptive Quasi-Newton with Low-Rank Approximation optimizer.

    This optimizer is designed for extremely ill-conditioned problems with
    off-diagonal Hessian elements. It combines diagonal preconditioning (like Adam)
    with a low-rank approximation of curvature information.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        rank (int, optional): rank of the curvature approximation (default: 1)
        clip_threshold (float, optional): threshold for gradient clipping (default: 10.0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, rank=1, clip_threshold=10.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 1 <= rank:
            raise ValueError(f"Invalid rank value: {rank}")
        if not 0.0 <= clip_threshold:
            raise ValueError(f"Invalid clip_threshold value: {clip_threshold}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, rank=rank,
                        clip_threshold=clip_threshold)
        super(AQNLR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AQNLR, self).__setstate__(state)

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
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get gradients and parameters
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError('AQNLR does not support sparse gradients')

                # Get optimizer state
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Initialize low-rank approximation components
                    rank = group['rank']
                    state['U'] = torch.zeros(p.numel(), rank, dtype=p.dtype, device=p.device)
                    state['V'] = torch.zeros(p.numel(), rank, dtype=p.dtype, device=p.device)
                    # Previous gradient and parameters
                    state['prev_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['prev_params'] = p.clone().detach()

                # Get optimizer hyperparameters
                beta1, beta2 = group['betas']
                lr = group['lr']
                eps = group['eps']
                weight_decay = group['weight_decay']
                clip_threshold = group['clip_threshold']
                rank = group['rank']

                # Update step count
                state['step'] += 1

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Reshape tensors for easier manipulation
                grad_flat = grad.view(-1)
                p_flat = p.view(-1)

                # Robust gradient clipping
                grad_norm = torch.norm(grad_flat)
                avg_precond = torch.mean(torch.sqrt(state['exp_avg_sq'].view(-1)) + eps)
                if grad_norm > clip_threshold * avg_precond:
                    grad_flat = grad_flat * (clip_threshold * avg_precond / grad_norm)

                # Update first moment estimate (momentum)
                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update second moment estimate (diagonal preconditioner)
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute bias-corrected first and second moment estimates
                exp_avg = state['exp_avg'].view(-1) / bias_correction1
                exp_avg_sq = state['exp_avg_sq'].view(-1) / bias_correction2

                # Update low-rank curvature approximation if we have previous gradient
                if state['step'] > 1:
                    # Compute parameter and gradient differences
                    s = p_flat - state['prev_params'].view(-1)
                    y_diff = grad_flat - state['prev_grad'].view(-1)

                    # Compute curvature pair importance
                    sy = torch.dot(s, y_diff)

                    # Only update if curvature is positive and significant
                    if sy > eps * torch.norm(s) * torch.norm(y_diff):
                        # Normalize s and y_diff
                        s_norm = s / torch.sqrt(sy)
                        y_diff_norm = y_diff / torch.sqrt(sy)

                        # Update low-rank approximation
                        # For rank-1, we just replace the vectors
                        if rank == 1:
                            state['U'][:, 0] = y_diff_norm
                            state['V'][:, 0] = s_norm
                        else:
                            # For higher ranks, we use a cyclic update strategy
                            idx = (state['step'] - 2) % rank
                            state['U'][:, idx] = y_diff_norm
                            state['V'][:, idx] = s_norm

                # Store current values for next iteration
                state['prev_grad'] = grad.clone().detach()
                state['prev_params'] = p.clone().detach()

                # Compute the preconditioned gradient
                # First apply diagonal preconditioner
                D_inv_sqrt = 1.0 / (torch.sqrt(exp_avg_sq) + eps)
                precond_grad = exp_avg * D_inv_sqrt

                # Then apply low-rank correction if we have at least one update
                if state['step'] > 1:
                    # Apply D^(-1/2) to V
                    V_scaled = state['V'] * D_inv_sqrt.unsqueeze(1)

                    # Compute V_scaled^T * precond_grad
                    Vt_g = torch.mv(V_scaled.t(), precond_grad)

                    # Compute U * (V_scaled^T * precond_grad)
                    U_Vt_g = torch.mv(state['U'], Vt_g)

                    # Apply the correction
                    precond_grad = precond_grad - U_Vt_g

                # Update parameters
                p.data.add_(precond_grad.view_as(p), alpha=-lr)

        return loss
