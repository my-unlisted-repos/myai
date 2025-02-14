import torch
from torch.optim import Optimizer

class Arnoldi(Optimizer):
    def __init__(self, params, lr=0.01, m=5, epsilon=1e-8):
        if m < 1: raise ValueError(f"Invalid m: {m}")
        defaults = dict(lr=lr, m=m, epsilon=epsilon)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['Q'] = []  # Orthonormal basis vectors

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            m = group['m']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Arnoldi does not support sparse gradients")

                state = self.state[p]
                Q = state['Q']

                # Flatten the gradient to handle as a vector
                grad_flatten = grad.clone().detach().flatten()
                v = grad_flatten.clone()
                alpha = []

                # Orthogonalize against existing basis vectors
                for q in Q:
                    q_flatten = q.flatten()
                    a = torch.dot(q_flatten, v)
                    alpha.append(a)
                    v.add_(-a * q_flatten)

                # Normalize the residual to get a new basis vector
                beta = torch.norm(v).item()
                if beta > epsilon:
                    q_new = (v / beta).view_as(grad)
                    Q.append(q_new.clone())
                    # Keep the basis size <= m
                    if len(Q) > m:
                        Q.pop(0)

                # Compute the step as the projection onto the current basis
                step = torch.zeros_like(grad)
                for i, q in enumerate(Q[:len(alpha)]):  # Handle case where Q grew
                    step.add_(alpha[i] * q)

                # Update parameters
                p.data.add_(-lr * step)

        return loss