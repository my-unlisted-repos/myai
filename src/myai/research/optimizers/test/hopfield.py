# pylint:disable=signature-differs, not-callable

import torch
import torch.nn as nn

class HopfieldSGD:
    """potentially interesting, or not, performance and LR similar to SGD."""
    def __init__(self, params, lr=0.01, memory_size=5, temperature=1.0):
        self.params = list(params)
        self.lr = lr
        self.memory_size = memory_size
        self.temperature = temperature
        self.memory = []
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad(): loss = closure()


        # Flatten and collect current gradients
        current_grads = [p.grad.clone() for p in self.params]
        current_vector = torch.cat([g.reshape(-1) for g in current_grads])

        # Add current gradient to memory (maintain fixed size)
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append(current_vector)

        # Compute attention scores using dot products
        if len(self.memory) > 0:
            mem_matrix = torch.stack(self.memory)
            similarities = torch.matmul(current_vector, mem_matrix.T) / self.temperature
            weights = torch.softmax(similarities, dim=0)

            # Compute weighted combination of memory vectors
            update_vector = torch.sum(weights[:, None] * mem_matrix, dim=0)

            # Reshape update vector to match parameters
            pointer = 0
            for p in self.params:
                numel = p.numel()
                update = update_vector[pointer:pointer+numel].reshape(p.grad.shape)
                p.add_(-self.lr * update)
                pointer += numel

        # Zero gradients after update
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

        self.step_count += 1

