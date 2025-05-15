from collections import deque
from operator import itemgetter
from typing import cast

import torch
from torch import Tensor

from torchzero.core import Chainable, Module, Vars, maybe_chain
from torchzero.linalg import matrix_inverse_root, sqrtmh
from torchzero.utils import TensorList, as_tensorlist


class Shampoo(Module):
    """A liquid preparation for washing the hair."""
    def __init__(
        self,
        eps: float = 1e-8,
        matrix_eps: float = 1e-5,
        update_freq: int = 1,
        exponent: int = 4,
        precondition_1d = True,
        beta: float | None = None,
        decay: float | None = None,
        inner: Chainable | None = None,
    ):
        defaults = dict(
            beta=beta,
            decay=decay,
            eps=eps,
            matrix_eps=matrix_eps,
            update_freq=update_freq,
            exponent=exponent,
            precondition_1d=precondition_1d,
        )
        super().__init__(defaults=defaults)
        self.global_state['step'] = 0

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad()
    def step(self, vars: Vars) -> Vars:
        self.global_state['step'] += 1
        step = self.global_state['step']
        update = vars.get_update()

        # update preconditioners
        for p, u in zip(vars.params, update):
            settings = self.settings[p]
            state = self.state[p]
            beta, decay, matrix_eps, update_freq, precondition_1d = itemgetter(
                'beta', 'decay', 'matrix_eps', 'update_freq', 'precondition_1d')(settings)

            # rmsprop/adagrad fallback
            if p.ndim == 0 or (p.ndim == 1 and not precondition_1d):
                if 'sq_grads' not in state:
                    state['sq_grads'] = torch.zeros_like(p)

                sq_grads = state['sq_grads']
                if decay is None: sq_grads.mul_(decay)

                if beta is None: sq_grads.add_(u)
                else: sq_grads.lerp_(u, 1-beta)

                continue

            # shampoo
            if 'preconditioners' not in state:
                state['preconditioners'] = [torch.eye(s, device=p.device, dtype=p.dtype) * matrix_eps for s in p.shape]
            preconditioners: list[Tensor] = state['preconditioners']

            if step % update_freq == 0:
                for dim in range(p.ndim):
                    if decay is not None: preconditioners[dim].mul_(decay)

                    mat = u.permute(*([dim] + [d for d in range(p.ndim) if d != dim])).contiguous()
                    rows = p.shape[dim]
                    cols = mat.numel() // rows
                    mat = mat.view(rows, cols)

                    stat_update = mat @ mat.T # rows, rows
                    stat_update = stat_update.to(preconditioners[dim].dtype)

                    if beta is None: preconditioners[dim].add_(stat_update)
                    else: preconditioners[dim].lerp_(stat_update, 1-beta)

        # step with inner module before applying preconditioner
        if self.children:
            inner_module = self.children['inner']
            inner_vars = inner_module.step(vars.clone(clone_update=False))
            vars.update_attrs_from_clone_(inner_vars)
            update = inner_vars.update
            assert update is not None

        # precondition
        preconditioned_update: list[Tensor] = []
        for p, u in zip(vars.params, update):
            settings = self.settings[p]
            state = self.state[p]
            exponent, eps, matrix_eps, precondition_1d= itemgetter(
                'exponent', 'eps', 'matrix_eps', 'precondition_1d')(settings)

            # rmsprop/adagrad fallback
            if p.ndim == 0 or (p.ndim == 1 and not precondition_1d):
                sq_grads = state['sq_grads']
                u.div_(sq_grads + eps)
                preconditioned_update.append(u)
                continue

            # shampoo
            precond_u = u
            preconditioners: list[Tensor] = state['preconditioners']
            for dim in range(p.ndim):
                inv_root = matrix_inverse_root(preconditioners[dim], exponent, matrix_eps)

                # dim_size, other_sizes
                permute_dims = [dim] + [d for d in range(p.ndim) if d != dim]

                precond_u = precond_u.permute(*permute_dims).contiguous()
                shape = precond_u.shape
                rows = shape[0]
                cols = precond_u.numel() // rows
                precond_u = precond_u.view(rows, cols)
                precond_u = inv_root @ precond_u

                # reshape back and reverse permutation
                precond_u = precond_u.view(shape)
                precond_u = precond_u.permute(*(permute_dims.index(d) for d in range(p.ndim))).contiguous()

            preconditioned_update.append(precond_u)

        vars.update = preconditioned_update
        return vars
