# pylint:disable=signature-differs, not-callable

import torch

def tt_decomposition(tensor, rank):
    device = tensor.device
    dtype = tensor.dtype
    shape = tensor.shape
    n_dims = len(shape)
    cores = []

    if n_dims == 0:
        return []

    current_tensor = tensor.reshape(shape[0], -1)
    current_rank = 1

    for i in range(n_dims - 1):
        dim_i = shape[i]

        U, S, Vh = torch.linalg.svd(current_tensor, full_matrices=False)

        rank_trunc = min(rank, S.shape[0])
        U_trunc = U[:, :rank_trunc]
        S_trunc = S[:rank_trunc]
        Vh_trunc = Vh[:rank_trunc, :]

        core = U_trunc.reshape(current_rank, dim_i, rank_trunc).to(dtype).to(device)
        cores.append(core)

        current_tensor = torch.diag(S_trunc) @ Vh_trunc

        if i < n_dims - 2:
            next_dim = shape[i + 1]
            current_tensor = current_tensor.reshape(rank_trunc * next_dim, -1)
        current_rank = rank_trunc

    last_core = current_tensor.reshape(current_rank, shape[-1], 1).to(dtype).to(device)
    cores.append(last_core)

    return cores

def tt_reconstruct(cores):
    if not cores:
        return torch.tensor(0.0)
    res = cores[0]
    for core in cores[1:]:
        res = torch.einsum('...a,adb->...db', res, core)
    res = res.squeeze(0).squeeze(-1)
    return res

class TensorTrainSGD(torch.optim.Optimizer):
    """tensor train decomposition of gradients"""
    def __init__(self, params, lr, rank):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rank <= 0:
            raise ValueError(f"Rank must be positive: {rank}")
        defaults = dict(lr=lr, rank=rank)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            rank = group['rank']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                original_shape = grad.shape

                try:
                    cores = tt_decomposition(grad, rank)
                except Exception as e:
                    print("TT decomposition failed, skipping parameter update.")
                    p.sub_(grad, alpha=lr)
                    continue

                grad_approx = tt_reconstruct(cores)
                grad_approx = grad_approx.reshape(original_shape).to(grad.device).to(grad.dtype)

                p.data.sub_(grad_approx, alpha=lr)
        return loss