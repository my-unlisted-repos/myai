# pylint:disable=signature-differs, not-callable

import math

import torch
import torch.optim as optim


class KnotRegularization(optim.Optimizer):
    def __init__(self, params, lr=0.01, knot_coeff=0.1, num_projections=10,
                 threshold=0.1, temperature=0.05):
        defaults = dict(lr=lr, knot_coeff=knot_coeff,
                        num_projections=num_projections,
                        threshold=threshold, temperature=temperature)
        super().__init__(params, defaults)
        self.knot_coeff = knot_coeff
        self.num_projections = num_projections
        self.threshold = threshold
        self.temperature = temperature

    def step(self, closure=None):
        if closure is None:
            raise ValueError("KnotRegularization requires a closure")

        # Compute main loss
        model_loss = closure(False)

        # Compute knot regularization
        reg_loss = self._compute_knot_regularization()

        # Combine losses
        total_loss = model_loss + reg_loss

        # Zero gradients and backpropagate
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

        total_loss.backward()

        # Perform SGD step
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(-lr * p.grad.data)

        return total_loss

    def _compute_knot_regularization(self):
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p.flatten())
        all_params = torch.cat(params)
        n = all_params.numel()

        # Pad parameters to make divisible by 3
        remainder = (3 - (n % 3)) % 3
        if remainder:
            padded = torch.cat([all_params, torch.zeros(remainder, device=all_params.device)])
        else:
            padded = all_params

        # Reshape to 3D knot (N, 3)
        knot = padded.view(-1, 3)
        if knot.size(0) < 2:
            return torch.tensor(0.0, device=knot.device)

        # Compute average crossing number over random projections
        total_acn = 0.0
        for _ in range(self.num_projections):
            # Random projection matrix
            rand_proj = torch.randn(3, 2, device=knot.device)
            rand_proj /= torch.norm(rand_proj, dim=0, keepdim=True)

            # Project knot to 2D
            proj_2d = knot @ rand_proj

            # Compute crossings for this projection
            crossings = self._compute_projected_crossings(proj_2d)
            total_acn += crossings

        avg_acn = total_acn / self.num_projections
        return self.knot_coeff * avg_acn

    def _compute_projected_crossings(self, proj):
        n = proj.size(0)
        if n < 2:
            return torch.tensor(0.0, device=proj.device)

        edge_starts = proj[:-1]  # (n-1, 2)
        edge_ends = proj[1:]     # (n-1, 2)
        n_edges = n - 1

        # Expand dimensions for broadcasting
        edge_starts_i = edge_starts.unsqueeze(1)  # (n_edges, 1, 2)
        edge_ends_i = edge_ends.unsqueeze(1)
        edge_starts_j = edge_starts.unsqueeze(0)  # (1, n_edges, 2)
        edge_ends_j = edge_ends.unsqueeze(0)

        # Direction vectors
        u = edge_ends_i - edge_starts_i  # (n_edges, 1, 2)
        v = edge_ends_j - edge_starts_j  # (1, n_edges, 2)

        # Compute 2D cross product (scalar)
        cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]  # (n_edges, n_edges)
        parallel_mask = torch.abs(cross) < 1e-8  # (n_edges, n_edges)

        # Non-parallel contribution (assumed to cross)
        contrib_non_parallel = torch.ones_like(parallel_mask, dtype=torch.float32)

        # Parallel case: compute minimal endpoint distances
        a = edge_starts_i  # (n_edges, 1, 2)
        b = edge_ends_i
        c = edge_starts_j
        d = edge_ends_j

        # Compute all endpoint distances
        d_ac = torch.norm(a - c, dim=-1)
        d_ad = torch.norm(a - d, dim=-1)
        d_bc = torch.norm(b - c, dim=-1)
        d_bd = torch.norm(b - d, dim=-1)

        min_dist = torch.min(torch.min(d_ac, d_ad), torch.min(d_bc, d_bd))
        contrib_parallel = torch.sigmoid((self.threshold - min_dist) / self.temperature)

        # Combine contributions
        contributions = torch.where(parallel_mask, contrib_parallel, contrib_non_parallel)

        # Create mask for non-consecutive edges
        i_indices = torch.arange(n_edges, device=proj.device).unsqueeze(1)
        j_indices = torch.arange(n_edges, device=proj.device).unsqueeze(0)
        non_consecutive = (j_indices > i_indices + 1)
        upper_triangle = (j_indices > i_indices)
        valid_pairs = non_consecutive & upper_triangle

        # Apply mask and sum
        masked_contrib = contributions * valid_pairs
        return masked_contrib.sum()


class KnotOptimizer(optim.Optimizer):
    """bro what"""
    def __init__(self, params, lr=0.01, knot_temperature=0.1,
                 crossing_threshold=0.5, topo_coeff=0.5,
                 num_projections=5, momentum=0.9):
        defaults = dict(lr=lr, knot_temperature=knot_temperature,
                        crossing_threshold=crossing_threshold,
                        topo_coeff=topo_coeff, num_projections=num_projections,
                        momentum=momentum)
        super().__init__(params, defaults)
        self.knot_cache = {}

    def _params_to_knot(self, params):
        """Convert parameters to 3D knot with momentum-aware smoothing"""
        with torch.no_grad():
            flat_params = torch.cat([p.flatten() for p in params])
            n = len(flat_params)

            # Pad to make divisible by 3
            pad = (3 - (n % 3)) % 3
            if pad > 0:
                flat_params = torch.cat([flat_params, torch.zeros(pad, device=flat_params.device)])

            # Create base knot (minimum 2 points)
            min_length = 6  # Ensures at least 2 points (3*2=6 elements)
            if len(flat_params) < min_length:
                flat_params = torch.cat([flat_params, torch.zeros(min_length - len(flat_params),
                                 device=flat_params.device)])

            knot = flat_params.view(-1, 3)

            # Apply momentum-based smoothing only if we have enough points
            if len(knot) > 1:
                for i in range(1, len(knot)):
                    knot[i] = knot[i-1] * self.defaults['momentum'] + knot[i] * (1 - self.defaults['momentum'])

            return knot

    def _compute_topological_gradient(self, knot):
        """Compute gradient that reduces knot complexity using differentiable crossings"""
        n = len(knot)
        grad = torch.zeros_like(knot)

        # Early return for knots that can't have crossings
        if n < 4:  # Need at least 2 segments to potentially cross
            return grad

        # Create edge vectors with momentum smoothing
        edges = knot[1:] - knot[:-1]

        # Handle edge case for small number of edges
        if len(edges) == 0:
            return grad

        # Repeat last edge only if we have edges to work with
        edges = torch.cat([edges, edges[-1].unsqueeze(0)])  # Now safe

        # Random projection approach for gradient estimation
        for _ in range(self.defaults['num_projections']):
            # Random projection matrix
            proj = torch.randn(3, 2, device=knot.device)
            proj /= torch.norm(proj, dim=0, keepdim=True)

            # Project knot to 2D
            proj_knot = knot @ proj

            # Compute differentiable crossings
            with torch.enable_grad():
                proj_knot.requires_grad_(True)
                crossings = self._differentiable_crossings(proj_knot)
                (-crossings).backward()  # Gradient ASCENT on crossing reduction # type:ignore
                grad += proj_knot.grad @ proj.t()
                proj_knot.grad = None

        return grad / self.defaults['num_projections']

    def _differentiable_crossings(self, proj_knot):
        """Differentiable approximation of crossing number using sigmoid gates"""
        n = len(proj_knot)
        total = 0.0
        temperature = self.defaults['knot_temperature']
        threshold = self.defaults['crossing_threshold']

        for i in range(n):
            for j in range(i+2, n-1):
                # Line segments: p1->p2 and q1->q2
                p1, p2 = proj_knot[i], proj_knot[i+1]
                q1, q2 = proj_knot[j], proj_knot[j+1]

                # Vectorized crossing test
                A = p2 - p1
                B = q1 - p1
                C = q2 - p1

                cross1 = A[0]*B[1] - A[1]*B[0]
                cross2 = A[0]*C[1] - A[1]*C[0]
                sign_change = torch.sigmoid(-cross1*cross2/temperature)

                D = q2 - q1
                E = p1 - q1
                F = p2 - q1
                cross3 = D[0]*E[1] - D[1]*E[0]
                cross4 = D[0]*F[1] - D[1]*F[0]
                sign_change2 = torch.sigmoid(-cross3*cross4/temperature)

                overlap = sign_change * sign_change2
                total += overlap * torch.sigmoid((threshold - self._seg_distance(p1,p2,q1,q2))/temperature)

        return total

    def _seg_distance(self, a, b, c, d):
        """Differentiable minimum distance between line segments"""
        # Parametrize both segments
        s = (b - a).unsqueeze(0)
        t = (d - c).unsqueeze(0)

        # Solve for closest points
        A = torch.sum(s * s, dim=1)
        B = torch.sum(s * t, dim=1)
        C = torch.sum(t * t, dim=1)
        D = torch.sum(s * (a - c), dim=1)
        E = torch.sum(t * (a - c), dim=1)

        det = A * C - B * B
        s_numer = B * E - C * D
        t_numer = A * E - B * D

        s_param = torch.clamp(s_numer / det, 0, 1)
        t_param = torch.clamp(t_numer / det, 0, 1)

        closest_on_s = a + s_param.unsqueeze(1) * s
        closest_on_t = c + t_param.unsqueeze(1) * t

        return torch.norm(closest_on_s - closest_on_t, dim=1)

    @torch.no_grad
    def step(self, closure=None):
        if closure is None:
            raise ValueError("Requires closure for loss computation")

        # Get all parameters
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)
        device = params[0].device

        # Convert parameters to knot structure
        knot = self._params_to_knot(params)

        # Compute topological gradient
        with torch.no_grad():
            topo_grad = self._compute_topological_gradient(knot)

        # Compute main loss gradient
        with torch.enable_grad(): loss = closure()
        # loss.backward()

        # Morph parameter gradients with topological information
        grad_idx = 0
        original_shapes = [p.shape for p in params]
        flat_grad = torch.cat([p.grad.flatten() for p in params])

        # Combine gradients
        combined_grad = (
            flat_grad +
            self.defaults['topo_coeff'] * topo_grad.flatten()[:len(flat_grad)]
        )

        # Distribute gradients back to parameters
        ptr = 0
        for p, shape in zip(params, original_shapes):
            numel = p.numel()
            p.grad = combined_grad[ptr:ptr+numel].view(shape)
            ptr += numel

        # Perform optimization step
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.add_(-lr * p.grad.nan_to_num(0,0,0))

        return loss