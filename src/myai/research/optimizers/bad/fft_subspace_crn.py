# pylint:disable=signature-differs, not-callable

import torch
import torch.fft
import numpy as np


class FFTSubspaceSCRN(torch.optim.Optimizer):
    """experiment with cubic regularized newton in an fft subspace"""
    def __init__(self, params, subspace_dim, reg_coef=1.0, momentum=0.9, damping=1e-6, max_iters=10, tol=1e-6, subspace='old'):
        params = list(params)
        self.params = params
        super().__init__(params, {})
        self.subspace_dim = subspace_dim
        self.reg_coef = reg_coef
        self.max_iters = max_iters
        self.tol = tol
        self.momentum = momentum
        self.damping = damping
        self.subspace = subspace

        self.momentum_buf = []

    def _compute_gradient(self, closure):
        loss = closure(False)
        grad = torch.autograd.grad(loss, self.params, create_graph=True)
        return grad

    def _fft_subspace(self, grad):
        grad_flat = torch.cat([g.view(-1) for g in grad])
        grad_fft = torch.fft.fft(grad_flat)
        magnitudes = torch.abs(grad_fft)
        top_freqs = torch.topk(magnitudes, self.subspace_dim).indices
        subspace_basis = torch.eye(len(grad_flat), device=grad_flat.device)[top_freqs]
        return subspace_basis

    def _new_fft_subspace(self, grad):
        """Improved FFT subspace with frequency windowing and momentum"""
        if len(self.momentum_buf) != len(grad):
            self.momentum_buf = [torch.zeros_like(g) for g in grad]

        # Combine current gradient with momentum
        grad_with_momentum = [
            g + self.momentum * buf
            for g, buf in zip(grad, self.momentum_buf)
        ]

        # Update momentum buffer
        for buf, g in zip(self.momentum_buf, grad):
            buf.mul_(self.momentum).add_(g, alpha=1 - self.momentum)

        # Convert to real-valued signal for FFT analysis
        grad_flat = torch.cat([g.view(-1) for g in grad_with_momentum])
        n = grad_flat.numel()

        # Apply windowing function to reduce spectral leakage
        window = torch.hann_window(n, device=grad_flat.device)
        windowed_grad = grad_flat * window

        # Multi-scale FFT analysis
        fft_result = torch.fft.rfft(windowed_grad)
        freqs = torch.fft.rfftfreq(n, d=1.0)
        magnitudes = torch.abs(fft_result)

        # Select dominant frequencies using energy-based threshold
        energy = magnitudes.pow(2).cumsum(-1)
        total_energy = energy[-1]
        threshold_idx = (energy > 0.9 * total_energy).nonzero()[0,0]
        dominant_freqs = freqs[:threshold_idx+1]

        # Create basis vectors using complex exponentials
        t = torch.linspace(0, 1, n, device=grad_flat.device)
        basis = []
        for freq in dominant_freqs[:self.subspace_dim//2]:
            basis.append(torch.sin(2 * np.pi * freq * t))
            basis.append(torch.cos(2 * np.pi * freq * t))

        # Pad with random directions if needed
        if len(basis) < self.subspace_dim:
            extra = torch.randn(self.subspace_dim - len(basis), n,
                              device=grad_flat.device)
            basis += [v / v.norm() for v in extra]

        # Orthonormalize basis
        basis_matrix = torch.stack(basis[:self.subspace_dim])
        q, _ = torch.linalg.qr(basis_matrix.T, mode='reduced')
        return q.T  # [subspace_dim x n]

    def _cubic_newton_step(self, grad_flat, hessian_vec_prod, subspace_basis):
        grad_sub = torch.matmul(subspace_basis, grad_flat)
        k = subspace_basis.size(0)
        hessian_sub = torch.zeros((k, k), device=grad_flat.device)

        # Construct Hessian in the subspace
        for j in range(k):
            e_j_sub = torch.zeros(k, device=grad_flat.device)
            e_j_sub[j] = 1.0
            hessian_sub[:, j] = hessian_vec_prod(e_j_sub)

        delta = self._solve_cubic_subproblem(grad_sub, hessian_sub)
        delta_full = torch.matmul(subspace_basis.t(), delta)
        return delta_full

    def _solve_cubic_subproblem(self, grad, hessian):
        """
        Solve the cubic regularized subproblem with added diagonal damping
        """
        delta = torch.zeros_like(grad)
        eye_matrix = torch.eye(len(grad), device=grad.device)

        for _ in range(self.max_iters):
            current_norm = torch.norm(delta)

            # Add multiple regularization components
            H = (
                hessian
                + self.reg_coef * current_norm * eye_matrix
                + 1e-6 * eye_matrix  # Additional diagonal damping
            )

            g = grad + torch.matmul(hessian + self.reg_coef * current_norm * eye_matrix, delta)

            try:
                d = torch.linalg.solve(H, -g)
            except torch.linalg.LinAlgError:
                # Fallback to pseudo-inverse if matrix is still singular
                d = torch.matmul(torch.linalg.pinv(H), -g)

            delta_new = delta + d
            if torch.norm(delta_new - delta) < self.tol:
                break

            delta = delta_new

        return delta

    def step(self, closure):
        grad = self._compute_gradient(closure)
        grad_flat = torch.cat([g.view(-1) for g in grad])
        if self.subspace == 'new': subspace_basis = self._new_fft_subspace(grad)
        else: subspace_basis = self._fft_subspace(grad)

        # Define Hessian-vector product in the subspace
        def hessian_vec_prod(v_sub):
            v = torch.matmul(subspace_basis.t(), v_sub)
            hv = torch.autograd.grad(grad_flat, self.params, grad_outputs=v, retain_graph=True)
            hv_flat = torch.cat([h.view(-1) for h in hv])
            return torch.matmul(subspace_basis, hv_flat)

        delta = self._cubic_newton_step(grad_flat, hessian_vec_prod, subspace_basis)
        # print(delta)
        idx = 0
        for p in self.params:
            p_size = p.numel()
            p.data.add_(delta[idx:idx + p_size].view(p.shape))
            idx += p_size


if __name__ == '__main__':
    # Example usage:
    # Define a simple model and loss function
    model = torch.nn.Linear(10, 1)
    optimizer = FFTSubspaceSCRN(model.parameters(), subspace_dim=5)

    # Define a closure for the loss function
    def cclosure(backward=True):
        output = model(torch.randn(32, 10))
        loss = torch.nn.functional.mse_loss(output, torch.randn(32, 1))
        if backward:
            optimizer.zero_grad()
            loss.backward()
        return loss

    # Perform an optimization step
    optimizer.step(cclosure)