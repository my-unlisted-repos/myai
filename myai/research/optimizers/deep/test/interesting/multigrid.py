import torch
from torch.optim import Optimizer

class MultigridSGD(Optimizer):
    def __init__(self, params, lr=0.1, scale=2.0, levels=3):
        defaults = dict(lr=lr, scale=scale, levels=levels)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            scale = group['scale']
            levels = group['levels']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("MultigridOptimizer does not support sparse gradients")

                # Flatten gradient to 1D for processing
                grad_flat = grad.flatten()
                n = grad_flat.numel()
                total_update = torch.zeros_like(grad_flat)

                for l in range(levels):
                    block_size = 2 ** l
                    alpha_l = lr * (scale ** l)

                    # Pad the gradient to be divisible by block_size
                    num_blocks = (n + block_size - 1) // block_size
                    padded_size = num_blocks * block_size
                    pad = padded_size - n
                    padded_grad = torch.nn.functional.pad(grad_flat, (0, pad))

                    # Reshape into blocks and average
                    blocked_grad = padded_grad.view(num_blocks, block_size)
                    coarse_grad = blocked_grad.mean(dim=1)

                    # Prolongate by repeating each coarse element block_size times
                    prolonged = coarse_grad.repeat_interleave(block_size)
                    # Truncate to original length (before padding)
                    prolonged = prolonged[:n]

                    # Accumulate the update
                    total_update += alpha_l * prolonged

                # Reshape the total update to match the parameter dimensions
                update = total_update.view(grad.shape)
                p.data.sub_(update)

        return loss