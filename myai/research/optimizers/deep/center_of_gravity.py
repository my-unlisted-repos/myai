import torch
from torch.optim import Optimizer

class CenterOfGravityMethod(Optimizer):
    def __init__(self, params, initial_bounds):
        """
        Only for 1-3 variables. This is very computationally expensive
        but has fast convergence assuming this implementation is correct, and it seems to work on booth function.

        Also only for convex functions.

        Args:
            params (iterable): Iterable of parameters to optimize.
            initial_bounds (tuple): Initial bounds for each parameter (min, max).
        """
        defaults = dict(initial_bounds=initial_bounds)
        super().__init__(params, defaults)

        self.initial_bounds = initial_bounds
        for group in self.param_groups:
            params = group['params']
            if len(params) != 1:
                raise ValueError("Only one parameter tensor supported")
            param = params[0]
            dim = param.numel()
            if dim not in (1, 2, 3):
                raise ValueError("Only 1-3 variables supported")

            # Initialize constraints
            group['dim'] = dim
            group['A'], group['b'] = self._init_constraints(initial_bounds, param)


    def _init_constraints(self, initial_bounds, param):
        dim = param.numel()
        A = []
        b = []
        for i in range(dim):
            lower, upper = self.initial_bounds
            # x_i >= lower
            a = torch.zeros(dim, device=param.device, dtype=param.dtype)
            a[i] = -1.0
            A.append(a)
            b.append(-lower)
            # x_i <= upper
            a = torch.zeros(dim, device=param.device, dtype=param.dtype)
            a[i] = 1.0
            A.append(a)
            b.append(upper)
        return torch.stack(A) if A else torch.empty(0, dim), torch.tensor(b, device=param.device, dtype=param.dtype)

    @torch.no_grad
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            param = group['params'][0]
            dim = group['dim']
            current_centroid = param.clone().detach().view(-1)

            if param.grad is None:
                raise RuntimeError("Gradient not computed")
            g = param.grad.clone().detach().view(-1)

            # Add new constraint: g^T (x - current_centroid) <= 0
            new_A = g.clone().detach()
            new_b = torch.dot(g, current_centroid)
            group['A'] = torch.cat([group['A'], new_A.unsqueeze(0)], dim=0)
            group['b'] = torch.cat([group['b'], new_b.unsqueeze(0)])

            # Compute new centroid
            if dim == 1:
                new_centroid = self._centroid_1d(group['A'], group['b'])
            else:
                new_centroid = self._centroid_nd(group['A'], group['b'], dim, current_centroid)

            param.set_(new_centroid.view_as(param))

        return loss

    def _centroid_1d(self, A, b):
        A = A.squeeze(-1)
        lower = -torch.inf
        upper = torch.inf
        for a_i, b_i in zip(A, b):
            a_i = a_i.item()
            b_i = b_i.item()
            if a_i > 0:
                upper = min(upper, b_i / a_i)
            elif a_i < 0:
                lower = max(lower, b_i / a_i)
            else:
                if b_i < 0:
                    raise ValueError("Infeasible constraints")
        if lower > upper:
            raise ValueError("Infeasible constraints")
        return torch.tensor((lower + upper) / 2, dtype=b.dtype, device=b.device)

    def _centroid_nd(self, A, b, dim, initial_point, num_samples=1000, burnin=100):
        samples = []
        x = initial_point.clone().detach()

        # Check if initial point is feasible
        if not (torch.mv(A, x) <= b + 1e-6).all():
            x = self._find_feasible_point(A, b, dim)

        for _ in range(burnin + num_samples):
            d = torch.randn(dim, device=A.device, dtype=A.dtype)
            d /= torch.norm(d) + 1e-8

            # Calculate valid t range
            Ad = torch.mv(A, d)
            b_Ax = b - torch.mv(A, x)

            t_low = -torch.inf
            t_high = torch.inf

            for i in range(A.size(0)):
                denominator = Ad[i]
                numerator = b_Ax[i]

                if abs(denominator) < 1e-8:
                    continue

                t = numerator / denominator
                if denominator > 0:
                    t_high = min(t_high, t)
                else:
                    t_low = max(t_low, t)

            if t_low > t_high:
                continue  # No valid step in this direction

            t = torch.rand(1, device=A.device, dtype=A.dtype) * (t_high - t_low) + t_low
            x_new = x + t * d

            if (torch.mv(A, x_new) <= b + 1e-6).all():
                x = x_new
                if _ >= burnin:
                    samples.append(x.clone())

        if not samples:
            raise RuntimeError("Failed to sample from polytope")

        return torch.stack(samples).mean(dim=0)

    def _find_feasible_point(self, A, b, dim, max_attempts=1000):
        # Simple sampling to find a feasible point
        for _ in range(max_attempts):
            x = torch.rand(dim, device=A.device, dtype=A.dtype) * 20 - 10  # Random point in [-10, 10]^dim
            if (torch.mv(A, x) <= b).all():
                return x
        raise RuntimeError("Could not find feasible initial point")