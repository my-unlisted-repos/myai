# type:ignore # pylint:disable=signature-differs, not-callable
import torch
from torch.optim import Optimizer


def sinkhorn(C, epsilon, max_iters, device='cuda'):
    m = C.size(0)
    K = torch.exp(-C / epsilon).to(device)
    u = torch.ones(m, 1, device=device) / m
    K = K + 1e-8  # Prevent numerical instability
    for _ in range(max_iters):
        v = 1.0 / (torch.mm(K.t(), u) + 1e-8)
        u = 1.0 / (torch.mm(K, v) + 1e-8)
    P = u.view(-1, 1) * K * v.view(1, -1)
    return P

class OTAdam(Optimizer):
    def __init__(self, params, lr=1e-3, epsilon=0.1, sinkhorn_iters=10):
        defaults = dict(lr=lr, epsilon=epsilon, sinkhorn_iters=sinkhorn_iters)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            epsilon = group['epsilon']
            sinkhorn_iters = group['sinkhorn_iters']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.dim() < 2:
                    p.sub_(p.grad, alpha = lr)
                    continue  # Only handle 2D (linear) or 4D (conv) parameters

                # Reshape parameter and gradient into 2D (num_units, features)
                W = p.data.view(p.size(0), -1)
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('OTAdam does not support sparse gradients')

                grad_2d = grad.view(grad.size(0), -1)
                m = W.size(0)

                # Compute pairwise squared Euclidean cost matrix
                W_norm = torch.sum(W ** 2, dim=1, keepdim=True)
                C = W_norm - 2 * torch.mm(W, W.t()) + W_norm.t()
                C = C.detach()  # Detach to prevent backprop through Sinkhorn

                # Move tensors to the correct device
                device = p.device
                C = C.to(device)

                # Compute optimal transport plan
                P = sinkhorn(C, epsilon, sinkhorn_iters, device)

                # Adjust the gradient using the transport plan
                G_adj_2d = torch.mm(P, grad_2d.to(device))

                # Reshape adjusted gradient back to original dimensions
                G_adj = G_adj_2d.view_as(grad)

                # Update parameters
                p.add_(-lr * G_adj)
        return loss