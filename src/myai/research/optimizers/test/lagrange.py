import torch

class LagrangeOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, window_size=3):
        if not 1 <= window_size <= 4:
            raise ValueError("window_size must be between 1 and 4")
        defaults = dict(lr=lr, window_size=window_size)
        super().__init__(params, defaults)

        # Precomputed Lagrange coefficients for buffer sizes 1 to 4
        self.coeffs = {
            1: [1.0],
            2: [-1.0, 2.0],
            3: [1.0, -3.0, 3.0],
            4: [-1.0, 4.0, -6.0, 4.0]
        }

        # Initialize gradient buffers for each parameter
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['grad_buffer'] = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            window_size = group['window_size']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Initialize buffer if not present
                if 'grad_buffer' not in state:
                    state['grad_buffer'] = []
                buffer = state['grad_buffer']

                # Store the current gradient and maintain buffer size
                buffer.append(grad.clone())
                if len(buffer) > window_size:
                    buffer.pop(0)

                m = len(buffer)
                if m == 0:
                    continue  # Safeguard, should not happen

                # Retrieve coefficients based on current buffer size
                coeff = self.coeffs.get(m, None)
                if coeff is None:
                    extrap_grad = grad  # Fallback to current gradient
                else:
                    # Compute extrapolated gradient
                    extrap_grad = torch.zeros_like(grad)
                    for c, g in zip(coeff, buffer):
                        extrap_grad.add_(c * g, alpha=1.0)

                # Update parameters using extrapolated gradient
                p.data.add_(extrap_grad, alpha=-lr)

        return loss