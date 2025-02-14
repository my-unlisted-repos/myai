import torch
from torch.optim import Optimizer

class CubicSplineOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.lr = lr

    def step(self, closure):
        """Performs a single optimization step using cubic spline line search.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            raise ValueError("Closure is required for CubicSplineOptimizer")

        # Initial evaluation at current parameters
        with torch.enable_grad():
            loss0 = closure()
            # loss0.backward()

        # Capture current parameters and gradients
        params0 = []
        grads0 = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                params0.append(p.data.clone())
                grads0.append(p.grad.data.clone())
        self.zero_grad()

        # Compute proposed direction (negative gradient scaled by learning rate)
        direction = [ -self.lr * g for g in grads0 ]

        # Evaluate at proposed point (theta + direction)
        for p, delta in zip(self._get_params(), direction):
            p.data.add_(delta)

        # Capture loss and gradients at proposed point
        with torch.enable_grad():
            loss1 = closure()
            # loss1.backward()

        grads1 = [p.grad.data.clone() for p in self._get_params()]
        self.zero_grad()

        # Restore original parameters
        for p, param0 in zip(self._get_params(), params0):
            p.data.copy_(param0)

        # Compute directional derivatives at t=0 and t=1
        dir_deriv0 = sum( torch.sum(g * d) for g, d in zip(grads0, direction) )
        dir_deriv1 = sum( torch.sum(g * d) for g, d in zip(grads1, direction) )

        # Compute cubic coefficients
        A = loss1 - loss0 - dir_deriv0
        B = dir_deriv1 - dir_deriv0
        a_coeff = B - 2 * A
        b_coeff = 3 * A - B

        # Solve for optimal t in [0, 1]
        t = self._find_optimal_t(a_coeff, b_coeff, dir_deriv0, loss0, loss1)

        # Apply optimal step
        for p, delta in zip(self._get_params(), direction):
            p.data.add_(delta, alpha=t if isinstance(t, float) else t.item())

        return loss0

    def _get_params(self):
        """Returns all parameters across all groups."""
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)
        return params

    def _find_optimal_t(self, a, b, c, loss0, loss1):
        """Finds the optimal step size t using cubic spline minimization."""
        # Quadratic equation coefficients: 3a t² + 2b t + c = 0
        if a == 0:
            if b == 0:
                return 0.0  # No valid step
            t = -c / (2 * b) if b != 0 else 0.0
            t = torch.clamp(t, 0.0, 1.0)
        else:
            discriminant = (2*b)**2 - 4*3*a*c
            if discriminant < 0:
                return 0.5  # Fallback to midpoint
            sqrt_discriminant = torch.sqrt(discriminant)
            t1 = (-2*b + sqrt_discriminant) / (6*a)
            t2 = (-2*b - sqrt_discriminant) / (6*a)
            t_candidates = [t1, t2]
            valid_ts = [t for t in t_candidates if 0 <= t <= 1]
            valid_ts += [0.0, 1.0]  # Check endpoints

            # Evaluate loss at valid t values
            min_loss = float('inf')
            best_t = 0.0
            for t in valid_ts:
                current_loss = loss0 + c*t + b*t**2 + a*t**3
                if current_loss < min_loss:
                    min_loss = current_loss
                    best_t = t
            return best_t
        return t.item() if isinstance(t, torch.Tensor) else t