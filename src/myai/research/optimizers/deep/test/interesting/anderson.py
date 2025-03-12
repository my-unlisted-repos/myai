import torch
from torch.optim import Optimizer

class AndersonOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, beta=1.0, m=5):

        defaults = dict(lr=lr, beta=beta, m=m)
        super().__init__(params, defaults)

        for group in self.param_groups:
            group['params_history'] = [] # Store flattened parameter vectors
            group['grad_history'] = []   # Store flattened gradient vectors (parameter differences)

    def _get_flat_params(self, param_group):
        views = []
        for p in param_group['params']:
            if p.grad is None:
                continue
            view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0) if views else torch.tensor([])

    def _set_params_from_flat(self, param_group, flat_params):
        offset = 0
        for p in param_group['params']:
            if p.grad is None:
                continue
            numel = p.data.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
            offset += numel

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            m = group['m']
            params_history = group['params_history']
            grad_history = group['grad_history']

            current_params_flat = self._get_flat_params(group).clone()
            grad_updates = []
            for p in group['params']:
                if p.grad is not None:
                    grad_updates.append(p.grad.data.view(-1))
            current_grad_flat = torch.cat(grad_updates, 0) if grad_updates else torch.tensor([])

            # 1. Basic Gradient Update (store for AA history later)
            basic_update_flat = -lr * current_grad_flat

            # Apply basic gradient update to get 'next' parameters (before AA)
            next_params_flat_basic = current_params_flat + basic_update_flat

            # 2. Store history (before applying AA)
            params_history.append(current_params_flat.cpu()) # Store on CPU to save GPU memory if history is long
            grad_history.append(basic_update_flat.cpu())

            if len(params_history) > m:
                params_history.pop(0)
                grad_history.pop(0)

            # 3. Anderson Acceleration Calculation (if history is sufficient)
            if len(params_history) >= 2: # Need at least two past points to apply AA
                Delta_P = torch.stack([params_history[i+1] - params_history[i] for i in range(len(params_history)-1)]).T.float().to(current_params_flat.device)
                Delta_G = torch.stack([grad_history[i+1] - grad_history[i] for i in range(len(grad_history)-1)]).T.float().to(current_params_flat.device)
                current_grad_cpu = basic_update_flat.cpu().float()

                try:
                    # Solve least squares: Delta_P @ alpha = -current_grad_cpu
                    solution, residuals, rank, singular_values = torch.linalg.lstsq(Delta_P, -current_grad_cpu) # Solve on CPU if history is on CPU
                    alpha = solution.to(current_params_flat.device)

                    # Anderson Accelerated update
                    aa_update_flat = torch.zeros_like(basic_update_flat)
                    for i in range(len(alpha)):
                        aa_update_flat -= alpha[i] * grad_history[-1-i].to(current_params_flat.device) # Use history of basic updates

                    # Combined update (mixing with beta)
                    combined_update_flat = (1 - beta) * basic_update_flat + beta * aa_update_flat

                    # Apply the update
                    updated_params_flat = current_params_flat + combined_update_flat

                except torch.linalg.LinAlgError: # Handle potential SVD convergence issues (rare, but for robustness)
                    print("LinAlgError in lstsq, using basic gradient update.")
                    updated_params_flat = next_params_flat_basic # Fallback to basic update
                except RuntimeError as e: # Handle other runtime errors during lstsq
                    print(f"RuntimeError in lstsq: {e}, using basic gradient update.")
                    updated_params_flat = next_params_flat_basic # Fallback to basic update

            else: # Not enough history, use basic gradient update
                updated_params_flat = next_params_flat_basic

            # Set parameters back into model
            self._set_params_from_flat(group, updated_params_flat)

        return loss


class AndersonSGD(Optimizer):
    """
    Implements Anderson Accelerated Stochastic Gradient Descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        m (int): number of history steps for Anderson Acceleration
        beta (float): regularization coefficient for least-squares problem
    """
    def __init__(self, params, lr=1e-3, m=5, beta=1e-4):
        defaults = dict(lr=lr, m=m, beta=beta)
        super().__init__(params, defaults)

        # Initialize history for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['residual_history'] = []  # Stores past residuals (-lr * grad)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            m = group['m']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AndersonSGD does not support sparse gradients')

                state = self.state[p]

                # Compute current residual: -lr * grad
                residual = -lr * grad

                # Update residual history
                state['residual_history'].append(residual.clone())
                if len(state['residual_history']) > m + 1:
                    state['residual_history'].pop(0)

                # Check if we have enough history to apply Anderson Acceleration
                if len(state['residual_history']) >= m + 1:
                    # Compute differences df_i = residual[i] - residual[i-1] for i=1..m
                    df_list = []
                    for i in range(1, m + 1):
                        df = state['residual_history'][i] - state['residual_history'][i-1]
                        df_list.append(df)

                    # Target b = -residual[-1] = lr * grad_current
                    b = -state['residual_history'][-1]

                    # Compute FtF matrix and Ft_b vector
                    m_curr = len(df_list)
                    FtF = torch.zeros(m_curr, m_curr, device=p.device)
                    Ft_b = torch.zeros(m_curr, device=p.device)

                    for i in range(m_curr):
                        df_i = df_list[i]
                        Ft_b[i] = torch.sum(df_i * b)
                        for j in range(m_curr):
                            FtF[i, j] = torch.sum(df_i * df_list[j])

                    # Add regularization
                    FtF_reg = FtF + beta * torch.eye(m_curr, device=FtF.device)

                    # Solve least-squares problem FtF_reg * gamma = Ft_b
                    try:
                        gamma = torch.linalg.solve(FtF_reg, Ft_b)
                    except RuntimeError:
                        # Fallback to pseudoinverse if singular
                        gamma = torch.linalg.pinv(FtF_reg) @ Ft_b

                    # Compute combined gradient
                    current_grad = grad
                    combined_grad = current_grad.clone()
                    for i in range(m_curr):
                        # grad_{k-i-1} = residual_history[-(i+2)] / (-lr)
                        grad_i = state['residual_history'][-(i + 2)] / (-lr)
                        combined_grad += gamma[i] * (grad_i - current_grad)

                    # Update parameters
                    p.data.add_(-lr * combined_grad)
                else:
                    # Regular SGD step
                    p.data.add_(-lr, grad)

        return loss