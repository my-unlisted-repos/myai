from collections import defaultdict

import numpy as np
import torch
from torch.optim import Optimizer


class Arnoldi(Optimizer):
    """bro uses the Arnoldi iteration's orthogonalization process to maintain a low-dimensional subspace of the most significant gradient directions"""
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



class ArnoldiStochastic(Optimizer):
    """does horrible atrocities"""
    def __init__(self, params, lr=0.1, m=5, epsilon=1e-6, damping=1e-3):
        defaults = dict(lr=lr, m=m, epsilon=epsilon, damping=damping)
        super().__init__(params, defaults)
        self.state = defaultdict(dict)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['s'] = []
                state['y'] = []
                state['V'] = []
                state['H'] = None

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            m = group['m']
            epsilon = group['epsilon']
            damping = group['damping']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                state['step'] += 1

                # Initialize previous parameters and gradients
                if 'prev_param' not in state:
                    state['prev_param'] = torch.clone(p.data).detach()
                    state['prev_grad'] = torch.clone(grad).detach()
                    continue

                # Compute s and y
                s = p.data - state['prev_param']
                y = grad - state['prev_grad']
                state['s'].append(s)
                state['y'].append(y)

                # Keep only last m pairs
                if len(state['s']) > m:
                    state['s'].pop(0)
                    state['y'].pop(0)
                state['prev_param'] = torch.clone(p.data).detach()
                state['prev_grad'] = torch.clone(grad).detach()

                # Perform Arnoldi every m steps
                if len(state['s']) == m:
                    s_list = state['s']
                    y_list = state['y']

                    V = []
                    H = torch.zeros((m, m), device=p.device)

                    # Arnoldi iteration
                    for j in range(m):
                        if j == 0:
                            v = s_list[0] / (torch.norm(s_list[0]) + epsilon)
                        else:
                            # Approximate Hv using y_list
                            Hv = torch.zeros_like(y_list[0])
                            for k in range(j):
                                Hv += H[k, j-1] * y_list[k]
                            # Orthogonalize
                            for i in range(j):
                                H[i, j] = torch.dot(V[i].flatten(), Hv.flatten())
                                Hv -= H[i, j] * V[i]
                            H_norm = torch.norm(Hv) + epsilon
                            H[j, j] = H_norm
                            v = Hv / H_norm
                        V.append(v)

                    # Compute Rizz values
                    H_small = H[:m, :m] + damping * torch.eye(m, device=H.device)
                    Rizz_values = torch.linalg.eigvalsh(H_small)
                    scaling = 1.0 / (Rizz_values + epsilon)

                    # Precondition gradient
                    grad_proj = torch.zeros_like(grad)
                    for i in range(len(V)):
                        coeff = torch.dot(V[i].flatten(), grad.flatten())
                        grad_proj += scaling[i] * coeff * V[i]

                    # Update parameters
                    p.data.add_(-lr * grad_proj)

                    # Reset buffers
                    state['s'] = []
                    state['y'] = []
                    state['V'] = []
                    state['H'] = None
                else:
                    # SGD step if buffer not full
                    p.data.add_(-lr * grad)

        return loss