import math

import torch
from torch.optim.optimizer import Optimizer


class HNGD(Optimizer):
    """Historical Natural Gradient Descent (HNGD).

    This optimizer approximates the natural gradient using historical gradient information,
    combining the strengths of natural gradient methods with the memory efficiency of
    limited-memory approaches like L-BFGS.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        history_size (int, optional): number of historical gradients to store (default: 10)
        beta (float, optional): coefficient for computing running average of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0.1)
    """

    def __init__(self, params, lr=1e-3, history_size=10, beta=0.9, eps=1e-8,
                 weight_decay=0, dampening=0.1):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if history_size <= 0:
            raise ValueError(f"Invalid history size: {history_size}")
        if beta < 0.0 or beta >= 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening value: {dampening}")

        defaults = dict(lr=lr, history_size=history_size, beta=beta, eps=eps,
                        weight_decay=weight_decay, dampening=dampening)
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
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('HNGD does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Initialize gradient history
                    state['grad_history'] = []
                    # Initialize previous gradient
                    state['prev_grad'] = torch.zeros_like(grad)

                exp_avg_sq = state['exp_avg_sq']
                grad_history = state['grad_history']
                prev_grad = state['prev_grad']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Update running average of squared gradients
                exp_avg_sq.mul_(group['beta']).addcmul_(grad, grad, value=1 - group['beta'])

                # Compute gradient difference and add to history
                grad_diff = grad - prev_grad

                # Apply dampening to gradient difference
                dampened_diff = grad_diff * math.sqrt(group['dampening'] * state['step'])

                # Update gradient history
                if len(grad_history) >= group['history_size']:
                    grad_history.pop(0)  # Remove oldest gradient
                grad_history.append(dampened_diff.clone())

                # Save current gradient as previous for next iteration
                state['prev_grad'] = grad.clone()

                # Compute natural gradient approximation
                if state['step'] > 1:  # Need at least one historical gradient
                    # Compute diagonal preconditioner
                    D_inv = 1.0 / (exp_avg_sq.sqrt() + group['eps'])

                    # Compute natural gradient using historical information
                    nat_grad = self._compute_natural_gradient(grad, grad_history, D_inv)

                    # Update parameters
                    p.addcmul_(nat_grad, torch.ones_like(nat_grad), value=-group['lr'])
                else:
                    # Fall back to diagonal preconditioner for first step
                    D_inv = 1.0 / (exp_avg_sq.sqrt() + group['eps'])
                    p.addcmul_(grad, D_inv, value=-group['lr'])

        return loss

    def _compute_natural_gradient(self, grad, grad_history, D_inv):
        """Compute natural gradient using historical gradient information.

        Args:
            grad (Tensor): Current gradient
            grad_history (list): List of historical gradient differences
            D_inv (Tensor): Inverse of diagonal preconditioner

        Returns:
            Tensor: Approximated natural gradient
        """
        # Apply diagonal preconditioner to current gradient
        precond_grad = grad * D_inv

        # If history is empty, return diagonally preconditioned gradient
        if not grad_history:
            return precond_grad

        # Construct V matrix from historical gradients
        V = torch.stack(grad_history, dim=-1)

        # Apply diagonal preconditioner to V
        V_precond = V * D_inv.unsqueeze(-1)

        # Compute V^T * D^(-1) * V (small m x m matrix)
        VtV = torch.matmul(V.transpose(0, 1).reshape(-1, V.shape[0]), V_precond.reshape(V.shape[0], -1))

        # Add identity matrix for numerical stability
        m = VtV.shape[0]
        VtV.add_(torch.eye(m, device=VtV.device))

        # Compute (I + V^T * D^(-1) * V)^(-1)
        try:
            VtV_inv = torch.inverse(VtV)
        except RuntimeError:
            # Fallback if inverse fails
            return precond_grad

        # Compute V * (I + V^T * D^(-1) * V)^(-1) * V^T * D^(-1) * grad
        # First: V^T * D^(-1) * grad
        Vt_precond_grad = torch.matmul(V.transpose(0, 1).reshape(-1, V.shape[0]),
                                       (grad * D_inv).reshape(grad.shape[0], -1))

        # Then: (I + V^T * D^(-1) * V)^(-1) * V^T * D^(-1) * grad
        temp = torch.matmul(VtV_inv, Vt_precond_grad)

        # Finally: V * (I + V^T * D^(-1) * V)^(-1) * V^T * D^(-1) * grad
        correction = torch.matmul(V_precond.reshape(V.shape[0], -1), temp)

        # Natural gradient = D^(-1) * grad - correction
        nat_grad = precond_grad - correction.reshape(grad.shape)

        return nat_grad


class HNGD_SVD(Optimizer):
    """Historical Natural Gradient Descent (HNGD).

    This optimizer approximates the natural gradient using historical gradient information,
    combining the strengths of natural gradient methods with the memory efficiency of
    limited-memory approaches like L-BFGS.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        history_size (int, optional): number of historical gradients to store (default: 10)
        beta (float, optional): coefficient for computing running average of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0.1)
        use_adaptive_history (bool, optional): whether to use adaptive history management (default: False)
        trust_region_max_ratio (float, optional): maximum ratio for trust region constraint (default: 10.0)
        momentum (float, optional): momentum factor (default: 0)
    """

    def __init__(self, params, lr=1e-3, history_size=10, beta=0.9, eps=1e-8,
                 weight_decay=0, dampening=0.1, use_adaptive_history=False,
                 trust_region_max_ratio=10.0, momentum=0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if history_size <= 0:
            raise ValueError(f"Invalid history size: {history_size}")
        if beta < 0.0 or beta >= 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening value: {dampening}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(lr=lr, history_size=history_size, beta=beta, eps=eps,
                        weight_decay=weight_decay, dampening=dampening,
                        use_adaptive_history=use_adaptive_history,
                        trust_region_max_ratio=trust_region_max_ratio,
                        momentum=momentum)
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
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('HNGD does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Initialize gradient history
                    if group['use_adaptive_history']:
                        state['grad_history'] = []  # Will store (grad_diff, informativeness) tuples
                    else:
                        state['grad_history'] = []  # Will store grad_diff tensors
                    # Initialize previous gradient
                    state['prev_grad'] = torch.zeros_like(grad)
                    # Initialize momentum buffer if momentum > 0
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg_sq = state['exp_avg_sq']
                grad_history = state['grad_history']
                prev_grad = state['prev_grad']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Update running average of squared gradients
                exp_avg_sq.mul_(group['beta']).addcmul_(grad, grad, value=1 - group['beta'])

                # Compute gradient difference and add to history
                grad_diff = grad - prev_grad

                # Apply dampening to gradient difference
                dampened_diff = grad_diff * math.sqrt(group['dampening'] * state['step'])

                # Update gradient history
                if group['use_adaptive_history']:
                    self._update_adaptive_history(dampened_diff, grad_history, group['history_size'])
                else:
                    if len(grad_history) >= group['history_size']:
                        grad_history.pop(0)  # Remove oldest gradient
                    grad_history.append(dampened_diff.clone())

                # Save current gradient as previous for next iteration
                state['prev_grad'] = grad.clone()

                # Compute natural gradient approximation
                if state['step'] > 1:  # Need at least one historical gradient
                    # Compute diagonal preconditioner
                    D_inv = 1.0 / (exp_avg_sq.sqrt() + group['eps'])

                    # Compute natural gradient using historical information
                    nat_grad = self._compute_natural_gradient(grad, grad_history, D_inv,
                                                             group['use_adaptive_history'])

                    # Apply trust region constraint if enabled
                    if group['trust_region_max_ratio'] > 0:
                        nat_grad = self._apply_trust_region(nat_grad, grad, group['trust_region_max_ratio'])

                    # Apply momentum if enabled
                    if group['momentum'] > 0:
                        if 'momentum_buffer' in state:
                            momentum_buffer = state['momentum_buffer']
                            momentum_buffer.mul_(group['momentum']).add_(nat_grad)
                            nat_grad = momentum_buffer

                    # Update parameters
                    p.addcmul_(nat_grad, torch.ones_like(nat_grad), value=-group['lr'])
                else:
                    # Fall back to diagonal preconditioner for first step
                    D_inv = 1.0 / (exp_avg_sq.sqrt() + group['eps'])
                    p.addcmul_(grad, D_inv, value=-group['lr'])

        return loss

    def _update_adaptive_history(self, grad_diff, grad_history, max_history_size):
        """Update gradient history using adaptive selection based on informativeness.

        Args:
            grad_diff (Tensor): Current gradient difference
            grad_history (list): List of (grad_diff, informativeness) tuples
            max_history_size (int): Maximum history size
        """
        # Compute "informativeness" of the new gradient difference
        informativeness = torch.norm(grad_diff).item()

        if len(grad_history) < max_history_size:
            # If history is not full, simply add the new gradient
            grad_history.append((grad_diff.clone(), informativeness))
        else:
            # Find the least informative gradient in history
            min_info_idx = min(range(len(grad_history)), key=lambda i: grad_history[i][1])

            # Replace it if the new gradient is more informative
            if informativeness > grad_history[min_info_idx][1]:
                grad_history[min_info_idx] = (grad_diff.clone(), informativeness)

        # Sort history by informativeness (for more efficient pruning later)
        grad_history.sort(key=lambda x: x[1], reverse=True)

    def _apply_trust_region(self, nat_grad, grad, max_ratio):
        """Apply trust region constraint to natural gradient.

        Args:
            nat_grad (Tensor): Natural gradient
            grad (Tensor): Original gradient
            max_ratio (float): Maximum allowed ratio between natural and original gradient norms

        Returns:
            Tensor: Constrained natural gradient
        """
        # Compute the norm ratio between natural gradient and original gradient
        grad_norm = torch.norm(grad)
        nat_grad_norm = torch.norm(nat_grad)

        # Apply trust region constraint if the ratio exceeds the threshold
        if nat_grad_norm > max_ratio * grad_norm and grad_norm > 0:
            nat_grad = nat_grad * (max_ratio * grad_norm / nat_grad_norm)

        return nat_grad

    def _compute_natural_gradient(self, grad, grad_history, D_inv, use_adaptive_history):
        """Compute natural gradient using historical gradient information.

        Args:
            grad (Tensor): Current gradient
            grad_history (list): List of historical gradient differences or (grad_diff, informativeness) tuples
            D_inv (Tensor): Inverse of diagonal preconditioner
            use_adaptive_history (bool): Whether adaptive history management is used

        Returns:
            Tensor: Approximated natural gradient
        """
        # Apply diagonal preconditioner to current gradient
        precond_grad = grad * D_inv

        # If history is empty, return diagonally preconditioned gradient
        if not grad_history:
            return precond_grad

        # Construct V matrix from historical gradients
        if use_adaptive_history:
            V = torch.stack([item[0] for item in grad_history], dim=-1)
        else:
            V = torch.stack(grad_history, dim=-1)

        # Apply diagonal preconditioner to V
        V_precond = V * D_inv.unsqueeze(-1)

        # Compute V^T * D^(-1) * V (small m x m matrix)
        VtV = torch.matmul(V.transpose(0, 1).reshape(-1, V.shape[0]), V_precond.reshape(V.shape[0], -1))

        # Add identity matrix for numerical stability
        m = VtV.shape[0]
        VtV.add_(torch.eye(m, device=VtV.device))

        # Compute (I + V^T * D^(-1) * V)^(-1) using more stable SVD approach
        try:
            U, S, V_svd = torch.svd(VtV)
            # Apply a threshold to singular values for numerical stability
            S_inv = torch.where(S > 1e-6, 1.0 / S, torch.zeros_like(S))
            VtV_inv = V_svd @ torch.diag(S_inv) @ U.t()
        except RuntimeError:
            # Fallback if SVD fails
            try:
                VtV_inv = torch.inverse(VtV)
            except RuntimeError:
                # If all else fails, return diagonally preconditioned gradient
                return precond_grad

        # Compute V * (I + V^T * D^(-1) * V)^(-1) * V^T * D^(-1) * grad
        # First: V^T * D^(-1) * grad
        Vt_precond_grad = torch.matmul(V.transpose(0, 1).reshape(-1, V.shape[0]),
                                       (grad * D_inv).reshape(grad.shape[0], -1))

        # Then: (I + V^T * D^(-1) * V)^(-1) * V^T * D^(-1) * grad
        temp = torch.matmul(VtV_inv, Vt_precond_grad)

        # Finally: V * (I + V^T * D^(-1) * V)^(-1) * V^T * D^(-1) * grad
        correction = torch.matmul(V_precond.reshape(V.shape[0], -1), temp)

        # Natural gradient = D^(-1) * grad - correction
        nat_grad = precond_grad - correction.reshape(grad.shape)

        return nat_grad
