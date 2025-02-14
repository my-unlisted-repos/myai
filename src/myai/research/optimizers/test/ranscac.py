import torch
from torch.optim import Optimizer

class RANSAC(Optimizer):
    def __init__(self, params, lr=1e-3, ransac_iters=100, min_samples=2, inlier_threshold=0.1):

        defaults = dict(lr=lr, ransac_iters=ransac_iters, min_samples=min_samples, inlier_threshold=inlier_threshold)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            ransac_iters = group['ransac_iters']
            min_samples = group['min_samples']
            inlier_threshold = group['inlier_threshold']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad_flat = grad.view(-1)
                n_elements = grad_flat.size(0)

                if n_elements < min_samples:
                    continue

                best_inlier_count = 0
                best_inliers = None

                for _ in range(ransac_iters):
                    indices = torch.randperm(n_elements, device=grad.device)[:min_samples]
                    samples = grad_flat[indices]
                    current_mean = samples.mean()

                    diffs = torch.abs(grad_flat - current_mean)
                    inliers = diffs < inlier_threshold
                    inlier_count = inliers.sum().item()

                    if inlier_count > best_inlier_count or (inlier_count == best_inlier_count and best_inliers is None):
                        best_inlier_count = inlier_count
                        best_inliers = inliers
                        best_mean = current_mean.item()

                if best_inliers is not None:
                    inlier_values = grad_flat[best_inliers]
                    update = inlier_values.mean()
                else:
                    update = grad_flat.mean()

                p.add_(-lr * update)
        return loss