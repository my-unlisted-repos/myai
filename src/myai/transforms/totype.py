import numpy as np
import torch

def tonumpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray): return x
    return np.asarray(x)

def totensor(x, dtype = None, device = None):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype, copy=False)
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, copy=False)
    return torch.from_numpy(np.array(x)).to(device=device, dtype=dtype,)