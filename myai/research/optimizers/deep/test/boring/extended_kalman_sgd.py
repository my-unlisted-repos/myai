import torch
from torch.optim import Optimizer

class ExtendedKalmanSGD(Optimizer):
    """
    Implements the EKFStochastic optimizer, a novel approach inspired by the Extended Kalman Filter.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1e-3).
        beta (float, optional): EMA decay rate for gradient squared (default: 0.999).
        alpha (float, optional): Scaling factor for measurement noise (default: 0.1).
        Q (float, optional): Process noise added to covariance (default: 1e-6).
        epsilon (float, optional): Small constant for numerical stability (default: 1e-8).
    """

    def __init__(self, params, lr=1e-3, beta=0.999, alpha=0.1, Q=1e-6, epsilon=1e-8):

        defaults = dict(lr=lr, beta=beta, alpha=alpha, Q=Q, epsilon=epsilon)
        super().__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['v'] = torch.zeros_like(p.data)  # EMA of squared gradients
                state['P'] = torch.ones_like(p.data)   # Diagonal covariance matrix

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
            alpha = group['alpha']
            Q = group['Q']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('EKFStochastic does not support sparse gradients')

                state = self.state[p]
                state['step'] += 1
                v, P = state['v'], state['P']

                # Update EMA of squared gradients
                v.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                # Compute measurement noise covariance
                R = alpha * v + epsilon

                # Predict covariance with process noise
                P_pred = P + Q

                # Compute Kalman gain
                K = P_pred / (P_pred + R)

                # Parameter update
                p.data.sub_(lr * K * grad)

                # Update covariance estimate
                P_new = (P_pred * R) / (P_pred + R)
                state['P'] = P_new

        return loss