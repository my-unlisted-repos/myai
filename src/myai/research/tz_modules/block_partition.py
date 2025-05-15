import math

import numpy as np
import torch

from torchzero.core import Chainable
from torchzero.utils import vec_to_tensors
from torchzero.modules.optimizers.shampoo import _merge_small_dims
from torchzero.modules.projections import Projection



class BlockPartition(Projection):
    """splits parameters into blocks along the largest dimension"""
    def __init__(self, modules: Chainable, max_size: int, batched: bool, project_update=True, project_params=False, project_grad=False):
        defaults = dict(max_size=max_size, batched=batched)
        super().__init__(modules, project_update=project_update, project_params=project_params, project_grad=project_grad, defaults=defaults)


    @torch.no_grad
    def project(self, tensors, vars, current):
        partitioned = []
        for p,t in zip(vars.params, tensors):
            if t.numel() == 1:
                partitioned.append(t)
                continue

            settings = self.settings[p]
            max_size = settings['max_size']
            largest_dim_idx = np.argmax(t.shape).item()
            largest_dim = t.shape[largest_dim_idx]
            params_per_slice = int(np.prod(t.shape) / largest_dim)

            if largest_dim * params_per_slice < max_size:
                partitioned.append(t)
                continue

            state = self.state[p]
            state['dim'] = largest_dim_idx
            batched = settings['batched']
            num_slices = math.ceil((params_per_slice * largest_dim) / max_size)

            if batched:
                slice_size = math.ceil(largest_dim / num_slices)
                padding = slice_size * num_slices - largest_dim
                state['padding'] = padding
                if padding != 0:
                    pad_shape = list(t.shape)
                    pad_shape[largest_dim_idx] = padding
                    t = torch.cat([t, torch.zeros(pad_shape, device=t.device, dtype=t.dtype)], largest_dim_idx)

                # (*, largest, *) -> (slices, *, largest_chunks, *)
                t = t.movedim(largest_dim_idx, 0).view(num_slices, params_per_slice, *t.shape[1:]).movedim(1, largest_dim_idx+1)
                partitioned.append(t)

            else:
                partitioned.extend(t.chunk(num_slices, largest_dim_idx))




