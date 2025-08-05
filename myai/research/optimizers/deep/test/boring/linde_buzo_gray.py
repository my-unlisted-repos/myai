import torch
from torch.optim import Optimizer
import math

class LindeBuzoGray(Optimizer):
    def __init__(self, params, lr=1e-3, block_size=256, split_interval=100, perturb_epsilon=0.01, decay=0.9):
        defaults = dict(
            lr=lr,
            block_size=block_size,
            split_interval=split_interval,
            perturb_epsilon=perturb_epsilon,
            decay=decay
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            block_size = group['block_size']
            split_interval = group['split_interval']
            perturb_epsilon = group['perturb_epsilon']
            decay = group['decay']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError("LBGOptimizer does not support sparse gradients")

                # Flatten gradient and split into blocks
                flat_grad = grad.view(-1)
                num_elements = flat_grad.size(0)
                padded_size = math.ceil(num_elements / block_size) * block_size
                device = grad.device

                # Pad gradient with zeros if necessary
                padded_grad = torch.zeros(padded_size, device=device)
                padded_grad[:num_elements] = flat_grad
                blocks = padded_grad.view(-1, block_size)

                # Initialize state if not present
                state = self.state[param]
                if 'codebook' not in state:
                    initial_centroid = torch.mean(blocks, dim=0, keepdim=True)
                    state['codebook'] = initial_centroid
                    state['sum'] = initial_centroid.clone() * blocks.size(0)
                    state['count'] = torch.tensor([blocks.size(0)], device=device)
                    state['step'] = 0

                codebook = state['codebook']
                sum_ = state['sum']
                count_ = state['count']
                current_step = state['step']

                # Assign blocks to nearest centroids
                distances = torch.cdist(blocks, codebook)
                assignments = distances.argmin(dim=1)

                # Quantize gradients using codebook
                quantized_blocks = codebook[assignments]
                padded_grad_quantized = quantized_blocks.view(-1)
                flat_grad_quantized = padded_grad_quantized[:num_elements]
                grad.copy_(flat_grad_quantized.view_as(grad))

                # Update codebook statistics
                K = codebook.size(0)
                new_sum = torch.zeros_like(sum_)
                new_count = torch.zeros_like(count_)

                for k in range(K):
                    mask = (assignments == k)
                    if mask.any():
                        new_sum[k] = blocks[mask].sum(dim=0)
                        new_count[k] = mask.sum().float()

                # Apply exponential moving average
                sum_ = sum_ * decay + new_sum * (1 - decay)
                count_ = count_ * decay + new_count * (1 - decay)
                codebook = sum_ / (count_[:, None] + 1e-8)

                # Split codebook periodically
                if current_step > 0 and current_step % split_interval == 0:
                    new_centroids = []
                    for centroid in codebook:
                        perturbation = torch.randn_like(centroid) * perturb_epsilon
                        new_centroids.extend([centroid + perturbation, centroid - perturbation])
                    codebook = torch.stack(new_centroids)
                    sum_ = codebook.clone() * (count_.sum() / codebook.size(0))
                    count_ = torch.full((codebook.size(0), ), (count_.sum() / codebook.size(0)).item(), device=device)

                # Update state
                state['codebook'] = codebook
                state['sum'] = sum_
                state['count'] = count_
                state['step'] = current_step + 1

                # Perform SGD step with quantized gradient
                param.data.add_(-lr * grad)

        return loss