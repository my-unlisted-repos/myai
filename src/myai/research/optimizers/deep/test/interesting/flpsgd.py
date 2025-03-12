import torch
from torch.optim.optimizer import Optimizer

class FLPSGD(Optimizer):
    """
    Implements Fractional Laplacian Preconditioned SGD.

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        s (float): fractional exponent (default: 0.5)
        epsilon (float): numerical stability constant (default: 1e-8)
        beta (float): EMA decay rate for covariance estimation (default: 0.9)
    """

    def __init__(self, params, lr=0.001, s=0.5, epsilon=1e-8, beta=0.9):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f'Invalid beta: {beta}')
        defaults = dict(lr=lr, s=s, epsilon=epsilon, beta=beta)
        super().__init__(params, defaults)

        # Initialize covariance matrices
        for group in self.param_groups:
            for p in group['params']:
                if p.dim() >= 2:
                    self._init_param_state(p)

    def _init_param_state(self, p):
        state = self.state[p]
        state['step'] = 0

        # Get dimensions
        left_dim = p.size(0)
        right_dim = p.numel() // left_dim

        # Initialize covariance matrices
        state['left_cov'] = torch.eye(left_dim, device=p.device) * self.defaults['epsilon']
        state['right_cov'] = torch.eye(right_dim, device=p.device) * self.defaults['epsilon']

    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            s = group['s']
            epsilon = group['epsilon']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.dim() < 2:  # 1D params use vanilla SGD
                    p.add_(-lr * grad)
                    continue

                state = self.state[p]
                state['step'] += 1

                # Reshape gradient to matrix
                grad_matrix = grad.view(grad.size(0), -1)
                left_dim, right_dim = grad_matrix.shape

                # Update covariance matrices with EMA
                left_cov = state['left_cov']
                right_cov = state['right_cov']

                # Compute current covariance estimates
                new_left = grad_matrix @ grad_matrix.T
                new_right = grad_matrix.T @ grad_matrix

                # EMA update
                state['left_cov'] = beta * left_cov + (1 - beta) * new_left
                state['right_cov'] = beta * right_cov + (1 - beta) * new_right

                # Add epsilon for numerical stability
                left_cov = state['left_cov'] + epsilon * torch.eye(left_dim, device=grad.device)
                right_cov = state['right_cov'] + epsilon * torch.eye(right_dim, device=grad.device)

                # Compute fractional inverse using SVD
                def _fractional_inv(matrix, power):
                    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
                    return (U * (S + epsilon)**(-power)) @ Vt

                left_precon = _fractional_inv(left_cov, s)
                right_precon = _fractional_inv(right_cov, s)

                # Apply preconditioning
                preconditioned_grad = left_precon @ grad_matrix @ right_precon
                preconditioned_grad = preconditioned_grad.view(grad.shape)

                # Update parameters
                p.add_(-lr * preconditioned_grad)