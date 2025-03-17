import torch
from torch.optim import Optimizer

class StochasticCG(Optimizer):
    """
    Implements a novel stochastic nonlinear conjugate gradient algorithm with debiased EMAs and stabilization.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float, optional): Learning rate. Default: 1e-3.
        beta1 (float, optional): EMA decay rate for gradients. Default: 0.9.
        beta2 (float, optional): EMA decay rate for squared gradients. Default: 0.99.
        epsilon (float, optional): Term to improve numerical stability. Default: 1e-8.
        reset_interval (int, optional): Reset conjugate direction every 'reset_interval' steps. Default: 10.
        beta_growth_bound (float, optional): Bound on beta growth rate. Default: 2.
        beta_damping_steps (int, optional): if beta grows for more than beta_growth_bound, it doesns't grow instead for this many steps. Default: 5.
        tracked_beta_decay_bound (float, optional): Bound on decay rate of tracked beta value to avoid collapse. Default: 3.
        restart_growth_steps (int, optional): after this many large growth steps, beta is reset to 0. Default: 1000.
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, epsilon=1e-8, reset_interval=10, beta_growth_bound=2, beta_damping_steps = 5, tracked_beta_decay_bound = 3, restart_growth_steps = 1000,):

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, reset_interval=reset_interval, beta_growth_bound=beta_growth_bound, restart_growth_steps = restart_growth_steps, tracked_beta_decay_bound=tracked_beta_decay_bound, beta_damping_steps=beta_damping_steps)
        super().__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('step', 0)

        self.prev_beta = None
        self.beta_damping_steps = 0
        self.restart_growth_steps = restart_growth_steps

    @torch.no_grad
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        assert len(self.param_groups) == 1
        for group in self.param_groups:
            params_with_grad = []
            m_hats = []
            previous_m_hats = []
            vs = []

            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            epsilon = group['epsilon']
            beta_growth_bound = group['beta_growth_bound']
            tracked_beta_decay_bound = group['tracked_beta_decay_bound']
            beta_damping_steps = group['beta_damping_steps']
            reset_interval = group['reset_interval']
            restart_growth_steps = group['restart_growth_steps']
            group['step'] += 1
            t = group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('StochasticCG does not support sparse gradients')

                state = self.state[p]

                # Initialize state if necessary
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['d'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['previous_m_hat'] = None

                # Update EMA of gradients and squared gradients
                state['m'] = beta1 * state['m'] + (1 - beta1) * grad
                state['v'] = beta2 * state['v'] + (1 - beta2) * torch.square(grad)

                # Debiasing
                m_hat = state['m'] / (1 - beta1 ** t)
                v_hat = state['v'] / (1 - beta2 ** t)

                params_with_grad.append(p)
                m_hats.append(m_hat)
                previous_m_hats.append(state['previous_m_hat'])
                vs.append(v_hat)

            if not params_with_grad:
                continue

            # Compute beta (Fletcher-Reeves)
            if all(ph is not None for ph in previous_m_hats):
                numerator = sum(torch.dot(mh.flatten(), mh.flatten()) for mh in m_hats)
                denominator = sum(torch.dot(ph.flatten(), ph.flatten()) for ph in previous_m_hats)
                beta = numerator / (denominator + epsilon)
                if self.prev_beta is None: self.prev_beta = beta
                else:
                    beta = min(beta, self.prev_beta * beta_growth_bound)
                    if beta == self.prev_beta * beta_growth_bound:
                        self.restart_growth_steps += 1
                        if self.beta_damping_steps < beta_damping_steps:
                            beta = self.prev_beta
                            self.beta_damping_steps += 1
                        else:
                            self.beta_damping_steps = 0
                        if self.restart_growth_steps >= restart_growth_steps:
                            beta = 0
                    else:
                        self.beta_damping_steps = 0
                        self.restart_growth_steps = 0
                    self.prev_beta = max(beta, self.prev_beta / tracked_beta_decay_bound)
            else:
                beta = 0.0

            # Reset conjugate direction periodically
            if t % reset_interval == 0:
                beta = 0.0

            # Update parameters and directions
            for p, m_hat, v_hat, prev_mh in zip(params_with_grad, m_hats, vs, previous_m_hats):
                state = self.state[p]

                # Update conjugate direction
                state['d'] = -m_hat + beta * state['d']

                # Adaptive learning rate
                adaptive_lr = lr / (torch.sqrt(v_hat) + epsilon)

                # Parameter update
                p.data.add_(adaptive_lr * state['d'])

                # Save current m_hat for next iteration
                state['previous_m_hat'] = m_hat.clone()

        return loss




class StabilizedSNCG(Optimizer):
    def __init__(self, params, lr=1e-3, gamma=0.9, restart_interval=None,
                 clip_threshold=None, eps=1e-8):
        defaults = dict(lr=lr, gamma=gamma, restart_interval=restart_interval,
                        clip_threshold=clip_threshold, eps=eps)
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['ema_grad'] = torch.zeros_like(p.data)
                state['prev_ema_debiased'] = None
                state['search_dir'] = None

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            restart_interval = group['restart_interval']
            clip_threshold = group['clip_threshold']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Update iteration count
                state['step'] += 1
                step = state['step']

                # Update EMA of gradient
                ema_grad = state['ema_grad']
                ema_grad.mul_(1 - gamma).add_(grad, alpha=gamma)

                # Debiasing the EMA
                bias_correction = 1 - (gamma ** step)
                debiased_ema = ema_grad / (bias_correction + eps)

                # Compute beta using Polak-Ribière-Polyak (PRP) formula
                beta = 0.0
                if state['prev_ema_debiased'] is not None:
                    prev_debiased = state['prev_ema_debiased']
                    numerator = torch.dot(debiased_ema.flatten(),
                                        (debiased_ema - prev_debiased).flatten())
                    denominator = torch.norm(prev_debiased) ** 2 + eps
                    beta = numerator / denominator

                # Update search direction
                if state['search_dir'] is None:
                    search_dir = -debiased_ema
                else:
                    search_dir = -debiased_ema + beta * state['search_dir']

                # Apply restart strategy
                if restart_interval and (step % restart_interval == 0):
                    search_dir = -debiased_ema

                # Gradient clipping
                if clip_threshold is not None:
                    current_norm = torch.norm(search_dir)
                    if current_norm > clip_threshold:
                        search_dir.mul_(clip_threshold / (current_norm + eps))

                # Update parameters with adaptive step size
                current_ema_norm = torch.norm(debiased_ema) + eps
                adaptive_lr = lr / current_ema_norm  # Optional adaptation
                p.data.add_(search_dir, alpha=adaptive_lr)

                # Save current state for next iteration
                state['search_dir'] = search_dir.clone()
                state['prev_ema_debiased'] = debiased_ema.clone()

        return loss