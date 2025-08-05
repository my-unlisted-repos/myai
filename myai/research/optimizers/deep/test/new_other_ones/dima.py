import torch
from torch.optim.optimizer import Optimizer


# Define the DiMA Optimizer
class DiMA(Optimizer):
    """Implements Directional Momentum Adam (DiMA) optimizer."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), beta_d=0.99,
                 gamma=5.0, delta=0.1, eps=1e-8, eps_d=1e-12, weight_decay=0):
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            beta_d (float, optional): coefficient used for computing running average
                of parameter updates (valley direction) (default: 0.99)
            gamma (float, optional): Boosting factor for parallel component (default: 1.0 - no boost)
            delta (float, optional): Dampening factor for perpendicular component (default: 1.0 - no dampening)
            eps (float, optional): term added to the denominator to improve
                numerical stability in Adam part (default: 1e-8)
            eps_d (float, optional): term added to the denominator for direction
                normalization stability (default: 1e-12)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= eps_d:
            raise ValueError("Invalid epsilon_d value: {}".format(eps_d))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= beta_d < 1.0:
            raise ValueError("Invalid beta_d parameter: {}".format(beta_d))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not gamma >= 0:
            raise ValueError("Invalid gamma value: {}".format(gamma))
        if not delta >= 0:
            raise ValueError("Invalid delta value: {}".format(delta))
        if gamma == 1.0 and delta == 1.0:
            print("Warning: DiMA running with gamma=1 and delta=1, equivalent to Adam.")

        defaults = dict(lr=lr, betas=betas, beta_d=beta_d, gamma=gamma, delta=delta,
                        eps=eps, eps_d=eps_d, weight_decay=weight_decay)
        super(DiMA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DiMA, self).__setstate__(state)

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
            beta_d = group['beta_d']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            eps_d = group['eps_d']
            gamma = group['gamma']
            delta = group['delta']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('DiMA does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of parameter updates
                    state['exp_avg_delta'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous parameter value
                    state['prev_p'] = p.clone() # Store initial point

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg_delta = state['exp_avg_delta']
                prev_p = state['prev_p']

                state['step'] += 1
                step = state['step']

                # === Standard Adam Update Computations ===
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                denom = v_hat.sqrt().add_(eps)
                u_adam = m_hat / denom # Basic Adam update direction (before LR)

                # === Directional Modification ===
                if step > 1 : # Need at least one step to compute delta_x
                    # Calculate previous step delta
                    delta_x_prev = p.data - prev_p # relies on p.data reflecting step t-1 result before this update

                    # Update EMA of parameter updates (valley direction estimate)
                    exp_avg_delta.mul_(beta_d).add_(delta_x_prev, alpha=1 - beta_d)

                    # Bias correction for d - Optional but might help initially
                    bias_correction_d = 1 - beta_d ** (step - 1) # step-1 because it uses delta_x_prev
                    d_hat = exp_avg_delta / bias_correction_d
                    # Instead of bias correction, just let it warm up

                    d_norm = torch.linalg.norm(d_hat)

                    if d_norm > eps_d: # Avoid division by zero or near-zero
                        v_dir = d_hat / d_norm # Normalized valley direction

                        # Project u_adam onto v_dir
                        # Ensure shapes are compatible for dot product if necessary
                        u_adam_flat = u_adam.view(-1)
                        v_dir_flat = v_dir.view(-1)

                        dot_product = torch.dot(u_adam_flat, v_dir_flat)

                        u_parallel = (dot_product * v_dir_flat).view_as(u_adam)
                        u_perp = u_adam - u_parallel

                        # Combine boosted parallel and dampened perpendicular components
                        u_final = gamma * u_parallel + delta * u_perp
                    else:
                        # If direction estimate is zero/unstable, fallback to Adam update
                        u_final = u_adam
                else:
                    # First step, fallback to Adam update
                    u_final = u_adam

                # === Parameter Update ===
                # Store current p before modifying it, for next iteration's delta_x_prev
                state['prev_p'] = p.clone() # store p at step t before update

                # Apply final update step
                p.add_(u_final, alpha=-lr)


        return loss
