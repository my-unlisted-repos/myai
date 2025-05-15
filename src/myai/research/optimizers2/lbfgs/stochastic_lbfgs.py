import warnings
from collections import deque

import torch
from torch.optim.optimizer import Optimizer


# Utility functions for handling potentially complex parameter structures
def _flatten_params(params):
    """Flattens a list of tensors into a single vector."""
    flat_params = torch.cat([p.detach().reshape(-1) for p in params])
    return flat_params

def _flatten_grads(params):
    """Flattens gradients into a single vector."""
    flat_grads = torch.cat([p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p.detach().reshape(-1)) for p in params])
    return flat_grads

def _unflatten_vector(vector, params):
    """Unflattens a vector back into the structure of params."""
    current_index = 0
    unflattened = []
    for p in params:
        numel = p.numel()
        unflattened.append(vector[current_index : current_index + numel].reshape(p.shape))
        current_index += numel
    return unflattened

class StochasticLBFGS(Optimizer):
    """
    Implements a stochastic L-BFGS algorithm, adapted for mini-batch settings.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate (default: 1.0). L-BFGS methods
            often work well with lr=1.0, but might need tuning.
        history_size (int, optional): Size of the L-BFGS history (m). (default: 10)
        update_freq (int, optional): Update L-BFGS memory every `update_freq` steps
            using accumulated changes. (default: 10)
        ema_beta (float, optional): Decay factor for EMA estimates of s and y
             (0 < beta < 1). If None, uses simple averaging over update_freq. (default: None)
        curvature_eps (float, optional): Small value to check curvature y^T s > eps.
            (default: 1e-6)
        damping (float, optional): Small damping factor added to the diagonal of the
             implicit Hessian approximation for stability. (default: 1e-8)
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0)
    """
    def __init__(self,
                 params,
                 lr=1.0,
                 history_size=10,
                 update_freq=10,
                 ema_beta=None, # Alternative: use EMA for s/y estimates
                 curvature_eps=1e-6,
                 damping=1e-8,
                 weight_decay=0):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 <= history_size:
            raise ValueError("Invalid history size: {}".format(history_size))
        if not 1 <= update_freq:
            raise ValueError("Invalid update frequency: {}".format(update_freq))
        if ema_beta is not None and not (0.0 < ema_beta < 1.0):
             raise ValueError("Invalid ema_beta value: {}".format(ema_beta))
        if not 0.0 <= curvature_eps:
            raise ValueError("Invalid curvature epsilon: {}".format(curvature_eps))
        if not 0.0 <= damping:
            raise ValueError("Invalid damping: {}".format(damping))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, history_size=history_size, update_freq=update_freq,
                        ema_beta=ema_beta, curvature_eps=curvature_eps,
                        damping=damping, weight_decay=weight_decay)
        super(StochasticLBFGS, self).__init__(params, defaults)

        # Stochastic L-BFGS specific state initialization
        # Needs to be done per parameter group if options differ,
        # but typically they are shared. We'll manage state globally here
        # using the first parameter group's settings.
        if len(self.param_groups) > 1:
            warnings.warn("StochasticLBFGS optimizer uses global state for history and averaging "
                          "based on the first parameter group's options.")

        group = self.param_groups[0]
        self._params = []
        for group in self.param_groups:
             self._params.extend(group['params'])

        # Check if all parameters are on the same device
        if len(set(p.device for p in self._params)) > 1:
            raise ValueError("All parameters must be on the same device.")
        self._device = self._params[0].device if self._params else torch.device('cpu')

        # Global state (shared across parameter groups)
        self._state = {
            'step': 0,
            's_history': deque(maxlen=group['history_size']),
            'y_history': deque(maxlen=group['history_size']),
            'rho_history': deque(maxlen=group['history_size']),
            's_accum': None, # Accumulated step vector (x_k - x_{k-update_freq})
            'y_accum': None, # Accumulated grad difference (g_k - g_{k-update_freq})
            'ema_s': None,   # EMA estimate of s
            'ema_y': None,   # EMA estimate of y
            'prev_flat_params': None,
            'prev_flat_grad': None,
            'gamma': 1.0, # Initial Hessian scaling factor
        }

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Optional for stochastic methods, but gradient
                computation must have happened before calling step().
        """
        loss = None
        if closure is not None:
            # Note: In stochastic settings, re-evaluating the loss might use a
            # different mini-batch, which L-BFGS traditionally doesn't expect.
            # We primarily rely on the gradients already computed.
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0] # Use options from the first group
        lr = group['lr']
        update_freq = group['update_freq']
        history_size = group['history_size']
        ema_beta = group['ema_beta']
        curvature_eps = group['curvature_eps']
        damping = group['damping']
        weight_decay = group['weight_decay']

        # Get current flattened parameters and gradients
        current_flat_params = _flatten_params(self._params)
        current_flat_grad = _flatten_grads(self._params)

        # Apply weight decay (L2 penalty)
        if weight_decay != 0:
            current_flat_grad.add_(current_flat_params, alpha=weight_decay)

        # --- State Initialization and Update ---
        state = self._state
        step = state['step']

        if step == 0:
            # Initialize accumulators and previous states on the first step
            if ema_beta is None:
                state['s_accum'] = torch.zeros_like(current_flat_params)
                state['y_accum'] = torch.zeros_like(current_flat_grad)
            else:
                state['ema_s'] = torch.zeros_like(current_flat_params)
                state['ema_y'] = torch.zeros_like(current_flat_grad)
            state['prev_flat_params'] = current_flat_params.clone()
            state['prev_flat_grad'] = current_flat_grad.clone()

        # Calculate step s_k = x_k - x_{k-1} and gradient diff y_k = g_k - g_{k-1}
        s_k = current_flat_params.sub(state['prev_flat_params'])
        y_k = current_flat_grad.sub(state['prev_flat_grad'])

        # Update accumulators or EMA estimates
        if ema_beta is None:
            state['s_accum'].add_(s_k)
            state['y_accum'].add_(y_k)
        else:
            state['ema_s'].mul_(ema_beta).add_(s_k, alpha=1 - ema_beta)
            state['ema_y'].mul_(ema_beta).add_(y_k, alpha=1 - ema_beta)

        # --- Periodic L-BFGS Memory Update ---
        if (step + 1) % update_freq == 0 and step > 0:
            if ema_beta is None:
                s_avg, y_avg = state['s_accum'], state['y_accum']
            else:
                # Use EMA estimates, possibly adjusted for bias correction
                # For simplicity, we use the raw EMA values here.
                s_avg, y_avg = state['ema_s'], state['ema_y']

            # Check curvature condition: y_avg^T s_avg > eps
            # Add damping to y_avg before the dot product for stability
            y_avg_damped = y_avg + damping * s_avg
            curvature = torch.dot(y_avg_damped, s_avg)

            if curvature > curvature_eps:
                # Add pair to history
                if len(state['s_history']) == history_size:
                    state['s_history'].popleft()
                    state['y_history'].popleft()
                    state['rho_history'].popleft()

                rho = 1.0 / curvature
                state['s_history'].append(s_avg.clone())
                state['y_history'].append(y_avg_damped.clone()) # Store damped y
                state['rho_history'].append(rho)

                # Update initial Hessian scaling factor gamma using the latest accepted pair
                state['gamma'] = curvature / torch.dot(y_avg_damped, y_avg_damped)

                # print(f"Step {step+1}: Updated L-BFGS history. Curvature={curvature:.4e}, Rho={rho:.4e}, Gamma={state['gamma']:.4e}")

            else:
                 # print(f"Step {step+1}: Skipped L-BFGS update. Curvature condition not met ({curvature:.4e} <= {curvature_eps})")
                 pass


            # Reset accumulators if not using EMA
            if ema_beta is None:
                state['s_accum'].zero_()
                state['y_accum'].zero_()

        # --- L-BFGS Two-Loop Recursion to compute search direction ---
        q = current_flat_grad.clone() # q = g_k
        num_history = len(state['s_history'])
        alpha = torch.empty(num_history, device=self._device)

        # First loop (backward)
        for i in range(num_history - 1, -1, -1):
            rho_i = state['rho_history'][i]
            s_i = state['s_history'][i]
            y_i = state['y_history'][i]
            alpha[i] = rho_i * torch.dot(s_i, q)
            q.add_(y_i, alpha=-alpha[i])

        # Apply initial Hessian approximation H_0 (scaled identity)
        # H_0 = gamma * I
        z = q.mul(state['gamma'])

        # Second loop (forward)
        for i in range(num_history):
            rho_i = state['rho_history'][i]
            s_i = state['s_history'][i]
            y_i = state['y_history'][i]
            beta_i = rho_i * torch.dot(y_i, z)
            z.add_(s_i, alpha=alpha[i] - beta_i)

        # Final search direction p_k = -z = -H_k g_k
        search_direction = z.neg()

        # --- Parameter Update ---
        # No line search, just use learning rate lr
        # Update the actual parameters in place

        unflattened_p_k = _unflatten_vector(search_direction, self._params)
        for p, p_k in zip(self._params, unflattened_p_k):
            if p.grad is not None: # Only update params with gradients
                if num_history == 0: p_k = p_k.sign() * 0.1
                p.data.add_(p_k, alpha=lr)

        # --- Store current state for next iteration ---
        state['prev_flat_params'] = current_flat_params.clone() # Already updated in place
        state['prev_flat_grad'] = current_flat_grad.clone()     # Gradient used in this step
        state['step'] += 1

        return loss

    def get_state(self):
        # Return necessary state components if needed for saving/loading
        return {
            'step': self._state['step'],
            's_history': list(self._state['s_history']),
            'y_history': list(self._state['y_history']),
            'rho_history': list(self._state['rho_history']),
            's_accum': self._state['s_accum'],
            'y_accum': self._state['y_accum'],
            'ema_s': self._state['ema_s'],
            'ema_y': self._state['ema_y'],
            'prev_flat_params': self._state['prev_flat_params'],
            'prev_flat_grad': self._state['prev_flat_grad'],
            'gamma': self._state['gamma'],
        }

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Custom state loading
        custom_state = state_dict['custom_state']
        self._state['step'] = custom_state['step']
        self._state['s_history'] = deque(custom_state['s_history'], maxlen=self.param_groups[0]['history_size'])
        self._state['y_history'] = deque(custom_state['y_history'], maxlen=self.param_groups[0]['history_size'])
        self._state['rho_history'] = deque(custom_state['rho_history'], maxlen=self.param_groups[0]['history_size'])
        self._state['s_accum'] = custom_state['s_accum']
        self._state['y_accum'] = custom_state['y_accum']
        self._state['ema_s'] = custom_state['ema_s']
        self._state['ema_y'] = custom_state['ema_y']
        self._state['prev_flat_params'] = custom_state['prev_flat_params']
        self._state['prev_flat_grad'] = custom_state['prev_flat_grad']
        self._state['gamma'] = custom_state['gamma']

        # Ensure tensors are on the correct device
        for key in ['s_accum', 'y_accum', 'ema_s', 'ema_y', 'prev_flat_params', 'prev_flat_grad']:
            if self._state[key] is not None:
                self._state[key] = self._state[key].to(self._device)
        for hist in ['s_history', 'y_history']:
            self._state[hist] = deque([t.to(self._device) for t in self._state[hist]], maxlen=self.param_groups[0]['history_size'])
        self._state['rho_history'] = deque([t # rho is scalar, device doesn't matter as much but keep consistent
            for t in self._state['rho_history']], maxlen=self.param_groups[0]['history_size'])


    def state_dict(self):
        state_dict = super().state_dict()
        # Add custom state
        state_dict['custom_state'] = self.get_state()
        return state_dict