from collections import deque
import math
import torch
from torch.optim.optimizer import Optimizer

class RastriginGD(Optimizer):
    r"""Implements an optimizer that fits a Rastrigin-like model.

    Assumes the loss landscape for each parameter p can be approximated by:
        L(p) ≈ C + p^2 - A * cos(2 * pi * B * p)
    where the minimum is at p=0. It fits 'A' based on gradient history.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        B (float, optional): controls how good the optimizer is, higher values are better.
                             (default: 10.0)
        history_len (int, optional): How many past steps (param, grad pairs)
                                     to use for fitting A (default: 10)
        min_history (int, optional): Minimum history points required before
                                     attempting to fit A (default: 3)
        eps (float, optional): term added to the denominator of the least squares
                               fit for numerical stability (default: 1e-8)
        fallback_adam (bool, optional): If True, uses Adam update rule when
                                        history is insufficient. If False, uses
                                        simple SGD update rule. (default: False)
        betas (Tuple[float, float], optional): Coefficients used for computing
                                               running averages of gradient and its square
                                               if fallback_adam is True (default: (0.9, 0.999))
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0) -
                                        Handled by base class logic if > 0, but
                                        interacts potentially complexly with model fit.
    """

    def __init__(self, params, lr=1e-3, B=10., history_len=10, min_history=3,
                 eps=1e-8, fallback_adam=True, betas=(0.9, 0.999), weight_decay=0):

        defaults = dict(lr=lr, B=B, history_len=history_len, min_history=min_history,
                        eps=eps, fallback_adam=fallback_adam, betas=betas,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Initialize state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                # Store (param_value, grad_value) tuples
                state['history'] = deque(maxlen=group['history_len'])
                state['A_fitted'] = None # Store the fitted A value
                if group['fallback_adam']:
                     # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Optional for most optimizers.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): # Ensure closure runs with grad enabled
                loss = closure()

        two_pi_B = 2.0 * math.pi * self.defaults['B'] # Use group B later

        for group in self.param_groups:
            lr = group['lr']
            B = group['B']
            history_len = group['history_len']
            min_history = group['min_history']
            eps = group['eps']
            fallback_adam = group['fallback_adam']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']

            group_two_pi_B = 2.0 * math.pi * B

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RastriginFitOptimizer does not support sparse gradients')

                state = self.state[p]
                state['step'] += 1
                step = state['step']

                # Apply weight decay (decoupled variant) before history update if needed
                # Or standard L2: grad = grad.add(p, alpha=weight_decay) # Standard L2
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay) # Decoupled weight decay


                # --- Store current state in history ---
                # Clone to avoid modifying history when p changes later in the step
                current_p_val = p.detach().clone()
                current_grad_val = grad.detach().clone()
                state['history'].append((current_p_val, current_grad_val))

                use_model_grad = False
                # --- Fit model parameter A ---
                if len(state['history']) >= min_history:
                    try:
                        # Extract history tensors
                        p_hist = torch.stack([h[0] for h in state['history']]) # Shape: (hist_len, *p.shape)
                        grad_hist = torch.stack([h[1] for h in state['history']])# Shape: (hist_len, *p.shape)

                        # Calculate targets (y) and features (x) for least squares: y = A * x
                        # y = grad - 2*p
                        y = grad_hist - 2.0 * p_hist
                        # x = -2 * pi * B * sin(2 * pi * B * p)
                        x = -group_two_pi_B * torch.sin(group_two_pi_B * p_hist)

                        # Reshape for batch matrix multiplication if p is multi-dimensional
                        original_shape = p.shape
                        if p.dim() > 1:
                            p_hist = p_hist.view(history_len, -1)
                            grad_hist = grad_hist.view(history_len, -1)
                            y = y.view(history_len, -1)
                            x = x.view(history_len, -1)


                        # Calculate A = sum(x*y) / sum(x*x) element-wise for each parameter dim
                        # Sum over the history dimension (dim=0)
                        sum_xy = torch.sum(x * y, dim=0)
                        sum_xx = torch.sum(x * x, dim=0)

                        # Fit A
                        A_fitted_flat = sum_xy / (sum_xx + eps)

                        # Clamp A >= 0
                        A_fitted_flat = torch.clamp(A_fitted_flat, min=0.0)

                        # Reshape A back to original parameter shape (minus history dim)
                        state['A_fitted'] = A_fitted_flat.view(original_shape)
                        use_model_grad = True

                    except Exception as e:
                        # Handle potential errors during fitting (e.g., numerical issues)
                        # print(f"Warning: Fitting A failed for parameter {p.shape}, step {step}. Error: {e}")
                        state['A_fitted'] = None # Invalidate A if fitting fails
                        use_model_grad = False

                # --- Calculate update step ---
                if use_model_grad and state['A_fitted'] is not None:
                    # Use the gradient from the fitted Rastrigin model
                    A = state['A_fitted']
                    sin_term = torch.sin(group_two_pi_B * current_p_val)
                    model_grad = 2.0 * current_p_val - A * group_two_pi_B * sin_term

                    # Update rule: Simple gradient descent using the model's gradient
                    p.add_(model_grad, alpha=-lr)

                else:
                    # Fallback: History too short or fitting failed
                    if fallback_adam:
                        # Use Adam update rule
                        exp_avg = state['exp_avg']
                        exp_avg_sq = state['exp_avg_sq']

                        # Decay the first and second moment running average coefficient
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step

                        step_size = lr / bias_correction1

                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                        p.addcdiv_(exp_avg, denom, value=-step_size)
                    else:
                        # Use simple SGD update rule
                        p.add_(grad, alpha=-lr)


        return loss