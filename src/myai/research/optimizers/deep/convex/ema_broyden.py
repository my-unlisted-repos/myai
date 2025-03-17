import torch
from torch.optim import Optimizer

class EMABroyden(Optimizer):
    """this beats L-BFGS on the single convex task I am testing. absolutely horrible on all other tasks."""
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
    def step(self, closure): # pylint:disable=signature-differs
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
                p.add_(d_k, alpha=lr)


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

                if state['step'] > 1: # Broyden Update starts from step 2
                    if new_grad is None:
                        raise RuntimeError("Gradient should be re-evaluated or differentiable=True should be set for Broyden update.")

                    s_k = d_k.reshape(-1, 1) # s_k = x_{k+1} - x_k = d_k in direction-only step
                    y_k = (new_grad.data - grad).reshape(-1, 1) # y_k = g_{k+1} - g_k approximation with mini-batch grads

                    B_k_y_k = torch.matmul(B_k, y_k)
                    s_k_minus_Bk_yk = s_k - B_k_y_k
                    s_k_t_Bk_yk = torch.matmul(s_k.transpose(0, 1), y_k)

                    if torch.abs(s_k_t_Bk_yk) > 1e-8: # Avoid division by zero
                        delta_B_k = torch.matmul(s_k_minus_Bk_yk, s_k.transpose(0, 1)) / s_k_t_Bk_yk
                        state['B_k'].add_(delta_B_k)
                    else:
                        # print("hey idiot, broyden update skipped due to small denominator.")
                        pass

        return loss

