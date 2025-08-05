from typing import Literal

import torch

from ...core import Chainable, TensorwiseTransform
from ...utils.linalg import matrix_power_eigh
from ..functional import safe_scaling_

class TensorAdagrad(TensorwiseTransform):
    """GG^T is preconditioned by full-matrix adagrad, which can than be repeated, leading to tensor adagrad of arbitrary order. Float64 is recommended."""
    def __init__(
        self,
        order: int = 3,
        beta: float | None = None,
        decay: float | None = None,
        concat_params=True,
        update_freq=1,
        init: Literal["identity", "zeros", "ones", "GGT"] = "zeros",
        sqrt: bool = True,
        divide: bool = False,
        inner: Chainable | None = None,
    ):
        defaults = dict(order=order, beta=beta, decay=decay, init=init, sqrt=sqrt, divide=divide)
        super().__init__(defaults, concat_params=concat_params, update_freq=update_freq, inner=inner,)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, loss, state, setting):
        order = setting['order']
        decay = setting['decay']
        beta = setting['beta']
        init = setting['init']

        GGs = []
        cur = tensor
        for _ in range(order-1):
            cur = cur.ravel().clip(max=(torch.finfo(cur.dtype).max ** 0.5)/2) # prevents overflows
            GG = torch.outer(cur, cur)
            GGs.append(GG)
            cur = GG

        # ----------------------------- initialize state ----------------------------- #
        if 'GG_0' not in state:
            for i, GG in enumerate(GGs):
                key = f'GG_{i}'
                if init == 'identity': state[key] = torch.eye(GG.size(0), device=GG.device, dtype=GG.dtype)
                elif init == 'zeros': state[key] =  torch.zeros_like(GG)
                elif init == 'ones': state[key] = torch.ones_like(GG)
                elif init == 'GGT': state[key] = GG.clone()
                else: raise ValueError(init)

        # ---------------------------------- update ---------------------------------- #
        for i, GG in enumerate(GGs):
            key = f'GG_{i}'
            if decay is not None: state[key].mul_(decay)

            if beta is not None: state[key].lerp_(GG, 1-beta)
            else: state[key].add_(GG)

        state['i'] = state.get('i', 0) + 1 # number of GGTs in sum

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, loss, state, setting):
        sqrt = setting['sqrt']
        divide = setting['divide']

        i = 0
        while f'GG_{i}' in state:
            i += 1

        i -= 1
        GG: torch.Tensor = state[f'GG_{i}']

        while f'GG_{i-1}' in state:
            if divide: GG = GG/state.get('i', 1)
            GG_prev: torch.Tensor = state[f"GG_{i-1}"]

            # precondition GG_prev by GG and set as new GG
            try:
                if sqrt:
                    GG = (matrix_power_eigh(GG, -1/2) @ GG_prev.ravel()).view_as(GG_prev)
                else:
                    GG = torch.linalg.solve(GG, GG_prev.ravel()).view_as(GG_prev) # pylint:disable = not-callable
            except torch.linalg.LinAlgError:
                return safe_scaling_(tensor)

            i -= 1

        if tensor.numel() == 1:
            GG = GG.squeeze()
            if sqrt: return tensor / GG.sqrt()
            return tensor / GG

        try:
            if sqrt: B = matrix_power_eigh(GG, -1/2)
            else: return torch.linalg.solve(GG, tensor.ravel()).view_as(tensor) # pylint:disable = not-callable

        except torch.linalg.LinAlgError:
            return safe_scaling_(tensor)

        return (B @ tensor.ravel()).view_as(tensor)

