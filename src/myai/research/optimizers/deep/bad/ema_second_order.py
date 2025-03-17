# pylint:disable = signature-differs
import torch
from torch.optim import Optimizer

class EMABFGS(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, history_size=10):
        defaults = dict(lr=lr, beta=beta, history_size=history_size)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['ema_grad'] = torch.zeros_like(p.data) # Initialize EMA gradient
                state['B_k'] = None # Hessian approximation (inverse) - initialize later
                state['s_history'] = [] # For L-BFGS (if we implement L-BFGS later)
                state['y_history'] = [] # For L-BFGS (if we implement L-BFGS later)


    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                state['grad'] = grad
                lr = group['lr']
                beta = group['beta']

                state['step'] += 1

                # 1. Update EMA gradient
                state['ema_grad'].mul_(beta).add_(grad, alpha=1 - beta)
                ema_grad = state['ema_grad']

                # Initialize B_k in the first step
                if state['step'] == 1:
                    state['B_k'] = torch.eye(p.numel(), dtype=p.dtype, device=p.device)

                B_k = state['B_k']

                # 2. Compute search direction: d_k = -B_k * ema_grad
                ema_grad_flat = ema_grad.reshape(-1, 1) # Flatten gradient
                d_k_flat = -torch.matmul(B_k, ema_grad_flat)
                d_k = d_k_flat.reshape_as(p) # Reshape back

                # 3. Perform parameter update (no line search - constant LR)
                p.add_(d_k, alpha=lr)
                state['d_k'] = d_k

        # Get new gradient (for next iteration's EMA and BFGS update)
        # if group['differentiable']: # For functional API, re-eval loss to get new grad at new point
        with torch.enable_grad():
            loss = closure() # Re-evaluate loss to get new gradient

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                new_grad = p.grad
                B_k = state['B_k']
                grad = state['grad']
                d_k = state['d_k']
                if state['step'] > 1: # BFGS Update starts from step 2
                    s_k = d_k.reshape(-1, 1) # s_k = x_{k+1} - x_k = p.data - p.old_data (approx) = d_k in direction only step
                    y_k = (new_grad.data - grad).reshape(-1, 1) # y_k = g_{k+1} - g_k approximation using mini-batch gradients

                    # BFGS Update (using flattened vectors)
                    s_k_t = s_k.transpose(0, 1)
                    y_k_t = y_k.transpose(0, 1)

                    rho_k = 1.0 / torch.matmul(y_k_t, s_k)
                    I = torch.eye(B_k.size(0), dtype=B_k.dtype, device=B_k.device)
                    term1 = torch.matmul(torch.matmul(I - rho_k * torch.matmul(s_k, y_k_t), B_k), (I - rho_k * torch.matmul(y_k, s_k_t)))
                    term2 = rho_k * torch.matmul(s_k, s_k_t)
                    state['B_k'] = term1 + term2

        return loss


class EMASR1(Optimizer): # Implement EMASR1 similarly, focusing on SR1 update rule.
    def __init__(self, params, lr=1e-3, beta=0.9):
        defaults = dict(lr=lr, beta=beta)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['ema_grad'] = torch.zeros_like(p.data) # Initialize EMA gradient
                state['B_k'] = None # Hessian approximation (inverse) - initialize later

    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step."""
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                state['grad'] = grad
                lr = group['lr']
                beta = group['beta']

                state['step'] += 1

                # 1. Update EMA gradient
                state['ema_grad'].mul_(beta).add_(grad, alpha=1 - beta)
                ema_grad = state['ema_grad']

                # Initialize B_k in the first step
                if state['step'] == 1:
                    state['B_k'] = torch.eye(p.data.numel(), dtype=p.data.dtype, device=p.data.device)

                B_k = state['B_k']

                # 2. Compute search direction: d_k = -B_k * ema_grad
                ema_grad_flat = ema_grad.reshape(-1, 1) # Flatten gradient
                d_k_flat = -torch.matmul(B_k, ema_grad_flat)
                d_k = d_k_flat.reshape_as(p.data) # Reshape back
                state['d_k'] = d_k

                # 3. Perform parameter update (no line search - constant LR)
                p.data.add_(d_k, alpha=lr)

        # Get new gradient (for next iteration's EMA and SR1 update)
        with torch.enable_grad():
            loss = closure() # Re-evaluate loss to get new gradient

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                new_grad = p.grad
                B_k = state['B_k']
                grad = state['grad']
                d_k = state['d_k']
                if state['step'] > 1: # SR1 Update starts from step 2

                    s_k = d_k.reshape(-1, 1) # s_k = x_{k+1} - x_k = d_k in direction-only step
                    y_k = (new_grad.data - grad).reshape(-1, 1) # y_k = g_{k+1} - g_k approximation with mini-batch grads

                    # SR1 Update (using flattened vectors)
                    delta_Bk_yk_minus_Bk_sk = y_k - torch.matmul(B_k, s_k)
                    sk_transpose = s_k.transpose(0, 1)
                    denominator = torch.matmul(sk_transpose, delta_Bk_yk_minus_Bk_sk)

                    # Check denominator to avoid division by zero (or very small value)
                    if torch.abs(denominator) > 1e-8: # Add a small tolerance
                        delta_B_k = torch.matmul(delta_Bk_yk_minus_Bk_sk, delta_Bk_yk_minus_Bk_sk.transpose(0, 1)) / denominator
                        state['B_k'].add_(delta_B_k)
                    else:
                        pass
                        # print("Warning: SR1 update skipped due to small denominator.")


        return loss


class EMADFP(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, history_size=10):

        defaults = dict(lr=lr, beta=beta, history_size=history_size)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['ema_grad'] = torch.zeros_like(p.data) # Initialize EMA gradient
                state['B_k'] = None # Inverse Hessian approximation - initialize later
                state['s_history'] = [] # For L-DFP if we implement limited memory later
                state['y_history'] = [] # For L-DFP if we implement limited memory later

    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step using EMA-DFP."""
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                state['grad'] = grad
                lr = group['lr']
                beta = group['beta']

                state['step'] += 1

                # 1. Update EMA gradient
                state['ema_grad'].mul_(beta).add_(grad, alpha=1 - beta)
                ema_grad = state['ema_grad']

                # Initialize B_k in the first step
                if state['step'] == 1:
                    state['B_k'] = torch.eye(p.numel(), dtype=p.dtype, device=p.device)

                B_k = state['B_k']

                # 2. Compute search direction: d_k = -B_k * ema_grad
                ema_grad_flat = ema_grad.reshape(-1, 1) # Flatten gradient
                d_k_flat = -torch.matmul(B_k, ema_grad_flat)
                d_k = d_k_flat.reshape_as(p) # Reshape back
                state['d_k'] = d_k

                # 3. Perform parameter update (no line search - constant LR)
                p.add_(d_k, alpha=lr)

        # Get new gradient (for next iteration's EMA and Broyden update)
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                new_grad = p.grad
                grad = state['grad']
                B_k = state['B_k']
                d_k = state['d_k']
                if state['step'] > 1: # DFP Update starts from step 2
                    if new_grad is None:
                        raise RuntimeError("Gradient should be re-evaluated or differentiable=True should be set for DFP update.")

                    s_k = d_k.reshape(-1, 1) # s_k = x_{k+1} - x_k = d_k in direction-only step
                    y_k = (new_grad.data - grad).reshape(-1, 1) # y_k = g_{k+1} - g_k approximation with mini-batch grads

                    s_k_t = s_k.transpose(0, 1)
                    y_k_t = y_k.transpose(0, 1)

                    y_k_t_s_k = torch.matmul(y_k_t, s_k)
                    if y_k_t_s_k > 1e-8: # Check for positive curvature condition (numerically stable check)
                        term1 = torch.matmul(s_k, s_k_t) / y_k_t_s_k
                        B_k_y_k = torch.matmul(B_k, y_k)
                        term2_num = torch.matmul(B_k_y_k, B_k_y_k.transpose(0, 1))
                        term2_den = torch.matmul(y_k_t, B_k_y_k)
                        term2 = term2_num / term2_den
                        state['B_k'].add_(term1 - term2)
                    else:
                        # print("Warning: DFP update skipped due to non-positive curvature.")
                        pass

        return loss



class EMAMcCormick(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9):

        defaults = dict(lr=lr, beta=beta)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['ema_grad'] = torch.zeros_like(p.data) # Initialize EMA gradient
                state['B_k'] = None # Inverse Hessian approximation - initialize later


    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step using EMA-McCormick."""
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                state['grad'] = grad
                lr = group['lr']
                beta = group['beta']

                state['step'] += 1

                # 1. Update EMA gradient
                state['ema_grad'].mul_(beta).add_(grad, alpha=1 - beta)
                ema_grad = state['ema_grad']

                # Initialize B_k in the first step
                if state['step'] == 1:
                    state['B_k'] = torch.eye(p.data.numel(), dtype=p.data.dtype, device=p.data.device)

                B_k = state['B_k']

                # 2. Compute search direction: d_k = -B_k * ema_grad
                ema_grad_flat = ema_grad.reshape(-1, 1) # Flatten gradient
                d_k_flat = -torch.matmul(B_k, ema_grad_flat)
                d_k = d_k_flat.reshape_as(p.data) # Reshape back
                state['d_k'] = d_k

                # 3. Perform parameter update (no line search - constant LR)
                p.data.add_(d_k, alpha=lr)

        # Get new gradient (for next iteration's EMA and McCormick update)
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = state['grad']
                new_grad = p.grad
                d_k = state['d_k']
                B_k = state['B_k']

                if state['step'] > 1: # McCormick Update starts from step 2
                    if new_grad is None:
                        raise RuntimeError("Gradient should be re-evaluated or differentiable=True should be set for McCormick update.")

                    s_k = d_k.reshape(-1, 1) # s_k = x_{k+1} - x_k = d_k in direction-only step
                    y_k = (new_grad.data - grad).reshape(-1, 1) # y_k = g_{k+1} - g_k approximation with mini-batch grads

                    s_k_t = s_k.transpose(0, 1)
                    y_k_t = y_k.transpose(0, 1)
                    denominator = torch.matmul(s_k_t, y_k)

                    if torch.abs(denominator) > 1e-8: # Avoid division by zero
                        delta_B_k = (torch.matmul(s_k, s_k_t) / denominator) - B_k
                        state['B_k'].add_(delta_B_k)
                    else:
                        # print("Warning: McCormick update skipped due to small denominator.")
                        pass
        return loss

class EMAGreenstadt(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9):

        defaults = dict(lr=lr, beta=beta)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['ema_grad'] = torch.zeros_like(p.data) # Initialize EMA gradient
                state['B_k'] = None # Inverse Hessian approximation - initialize later


    @torch.no_grad
    def step(self, closure):
        """Performs a single optimization step using EMA-Greenstadt."""
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                state['grad'] = grad
                lr = group['lr']
                beta = group['beta']

                state['step'] += 1

                # 1. Update EMA gradient
                state['ema_grad'].mul_(beta).add_(grad, alpha=1 - beta)
                ema_grad = state['ema_grad']

                # Initialize B_k in the first step
                if state['step'] == 1:
                    state['B_k'] = torch.eye(p.data.numel(), dtype=p.data.dtype, device=p.data.device)

                B_k = state['B_k']

                # 2. Compute search direction: d_k = -B_k * ema_grad
                ema_grad_flat = ema_grad.reshape(-1, 1) # Flatten gradient
                d_k_flat = -torch.matmul(B_k, ema_grad_flat)
                d_k = d_k_flat.reshape_as(p.data) # Reshape back
                state['d_k'] = d_k

                # 3. Perform parameter update (no line search - constant LR)
                p.data.add_(d_k, alpha=lr)

                # Get new gradient (for next iteration's EMA and Greenstadt update)

        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                new_grad = p.grad
                grad = state['grad']
                B_k = state['B_k']
                d_k = state['d_k']

                if state['step'] > 1: # Greenstadt Update starts from step 2
                    if new_grad is None:
                        raise RuntimeError("Gradient should be re-evaluated or differentiable=True should be set for Greenstadt update.")

                    s_k = d_k.reshape(-1, 1) # s_k = x_{k+1} - x_k = d_k in direction-only step
                    y_k = (new_grad.data - grad).reshape(-1, 1) # y_k = g_{k+1} - g_k approximation with mini-batch grads

                    s_k_t = s_k.transpose(0, 1)
                    y_k_t = y_k.transpose(0, 1)
                    denominator = torch.matmul(s_k_t, y_k)

                    if torch.abs(denominator) > 1e-8: # Avoid division by zero
                        Hy_k = torch.matmul(B_k, y_k)
                        delta_B_k = (torch.matmul((s_k - Hy_k), y_k_t) + torch.matmul(y_k, (s_k - Hy_k).transpose(0, 1))) / denominator
                        state['B_k'].add_(delta_B_k)
                    else:
                        # print("Warning: Greenstadt update skipped due to small denominator.")
                        pass
        return loss