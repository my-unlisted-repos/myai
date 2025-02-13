# pylint:disable=signature-differs, not-callable
"""DeepSeek"""
import itertools
import random
from collections.abc import Callable

import numpy as np
import torch
import torchzero as tz
from torch import nn


def _raysearch_linesearch(closure, params: tz.TensorList, pvec, update, lr, loss, max_ls_iter, max_ars_iter):
    """this has everything you can possibly put in a line search. with accelerated random search if all of it failed. but there is a bug somewhere..."""
    pvec.sub_(update, alpha=lr)
    params.from_vec_(pvec)
    loss2 = closure(False)
    if loss2 < loss:
        return lr * 2

    niter = 0
    init_lr = lr
    cur_lr = lr

    loss2 = closure(False)
    while loss2 > loss:
        lr /= 1.5
        cur_lr /= 2
        pvec.add_(update, alpha = cur_lr)
        niter += 1
        params.from_vec_(pvec)
        loss2 = closure(False)

        if niter == max_ls_iter:
            pvec.add_(update, alpha = cur_lr + init_lr)
            cur_lr = init_lr
            params.from_vec_(pvec)
            loss2 = closure(False)
            niter = 0

            if loss2 > loss:
                while loss2 > loss:
                    cur_lr /= 2
                    pvec.sub_(update, alpha = cur_lr)
                    params.from_vec_(pvec)
                    loss2 = closure(False)
                    niter += 1

                    if niter >= max_ls_iter:
                        pvec.sub_(update, alpha = cur_lr)
                        lr = init_lr
                        cur_lr = init_lr

                        update = params.grad.to_vec()
                        std = update.std()
                        if 0 < std < 1:
                            update /= std + 1e-6

                        pvec.sub_(update, alpha = cur_lr)
                        params.from_vec_(pvec)
                        loss2 = closure(False)

                        niter = 0
                        while loss2 > loss:
                            cur_lr /= 2
                            pvec.add_(update, alpha = cur_lr)
                            params.from_vec_(pvec)
                            loss2 = closure(False)

                            niter += 1
                            if niter >= max_ls_iter:
                                pvec.add_(update, alpha = cur_lr)

                                cur_lr = init_lr * 2
                                for i in range(max_ars_iter):
                                    vec = params.sample_like(cur_lr)

                                    params.add_(vec)
                                    if loss2 < loss: break
                                    params.sub_(vec)
                                    cur_lr /= 2
                                else:
                                    lr = init_lr * 2
                                    vec = params.sample_like(params.mean().abs_().clamp_(1e-3).mul_(0.1))
                                    params.add_(vec)
                                break
                        break
                # opposite direction decreased loss after line search
                else: lr = init_lr / 1.5
            # opposite direction instantly decreased loss
            else:
                lr = init_lr * 2
            break

    return lr


class RaySubspace(tz.core.TensorListOptimizer):
    """estimates a newton step in a meaningful subspace which spans gradient, momentum, difference between those, etc.

    Args:
        params (_type_): _description_
        lr (_type_, optional): lr. Defaults to 1e-3.
        num_directions (int, optional): _description_. Defaults to 3.
        epsilon (_type_, optional): _description_. Defaults to 1e-5.
        momentum (float, optional): _description_. Defaults to 0.9.
        paramwise (bool, optional): _description_. Defaults to False.
    """
    def __init__(self, params, lr=1e-3, num_directions=10, epsilon=1e-5, momentum=0.9, max_ls_iter = 4, max_ars_iter = 20, paramwise=False):
        defaults = dict(lr=lr, num_directions=num_directions, epsilon=epsilon, momentum=momentum)
        super().__init__(params, defaults)

        self.max_ls_iter = max_ls_iter

        self.init_lr = lr
        self.lrs = []
        self.max_ars_iter = max_ars_iter

        self.prev_grad = None; self.velocity = None; self.prev_update = None
        self.paramwise = paramwise

    @torch.no_grad()
    def _global_step(self, closure):
        lr = self.defaults['lr']
        num_directions = self.get_first_group_key('num_directions')
        epsilon = self.get_first_group_key('epsilon')
        momentum = self.get_first_group_key('momentum')


        with torch.enable_grad(): loss = closure()
        params = self.get_params().with_grad()
        p = params.to_vec()
        grad = params.grad.to_vec()
        # Initialize state for this parameter

        if self.prev_grad is None: self.prev_grad = torch.zeros_like(grad)
        if self.velocity is None: self.velocity = torch.zeros_like(grad)

        # Compute velocity
        self.velocity += grad
        self.velocity *= momentum

        # Store previous gradient
        self.prev_grad.copy_(grad)

        # Select directions
        directions = self.select_directions(p, grad, self.prev_grad, self.velocity, self.prev_update, num_directions)

        # Estimate Hessian in the selected directions
        hessian_estimates = self.estimate_hessian_global(p, directions, epsilon, closure, params)

        # Compute update step
        update = self.compute_update(grad, directions, hessian_estimates)

        # Reshape update to match parameter shape and apply
        update = update.view(p.shape)

        lr = self.defaults['lr'] = _raysearch_linesearch(closure, params, p, update, lr, loss, self.max_ls_iter, self.max_ars_iter)
        self.prev_update = update

        self.lrs.append(lr)
        if len(self.lrs) > 10:
            del self.lrs[0]
            if np.max(self.lrs) < self.init_lr / 100:
                lr = self.defaults['lr'] = self.init_lr
                self.lrs[-1] = lr

        return loss

    @torch.no_grad()
    def _paramwise_step(self, closure):
        with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            num_directions = group['num_directions']
            epsilon = group['epsilon']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.flatten()

                # Initialize state for this parameter
                state = self.state[p]
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(grad)
                if 'prev_update' not in state:
                    state['prev_update'] = torch.zeros_like(grad)

                # Compute velocity
                state['velocity'] = momentum * state['velocity'] + grad

                # Store previous gradient
                prev_grad = state['prev_grad']
                state['prev_grad'] = grad.clone()

                # Select directions
                directions = self.select_directions(p, grad, prev_grad, state['velocity'], state['prev_update'], num_directions)

                # Estimate Hessian in the selected directions
                hessian_estimates = self.estimate_hessian_paramwise(p, directions, epsilon, closure)

                # Compute update step
                update = self.compute_update(grad, directions, hessian_estimates)

                # Reshape update to match parameter shape and apply
                update = update.view(p.shape)
                state['prev_update'] = update
                p.add_(update, alpha=-lr)
        return loss

    @torch.no_grad
    def step(self, closure):
        if self.paramwise: return self._paramwise_step(closure)
        return self._global_step(closure)

    @torch.no_grad
    def select_directions(self, param, grad, prev_grad, velocity, prev_update, num_directions):
        directions = []

        # 1. Gradient direction
        norm = torch.linalg.vector_norm(grad)
        if norm > 0:
            grad_norm = grad/norm
            directions.append(grad_norm)
        else: grad_norm = None

        # 2. Velocity direction
        norm = torch.linalg.vector_norm(velocity)
        if norm > 0:
            velocity_norm = velocity/norm
            directions.append(velocity_norm)
        else: velocity_norm = None

        # 3. Previous gradient direction
        norm = torch.linalg.vector_norm(prev_grad)
        if norm > 0:
            prev_grad_norm = prev_grad / norm
            directions.append(prev_grad_norm)
        else: prev_grad_norm = None

        # 4. Previous update direction
        if prev_update is not None:
            norm = torch.linalg.vector_norm(prev_update)
            if norm > 0:
                prev_update_norm = prev_update / norm
                directions.append(prev_update_norm)
            else:
                prev_update_norm = None
        else:
            prev_update_norm = None

        # 5. Param itself
        norm = torch.linalg.vector_norm(param)
        if norm > 0: directions.append(param / norm)

        # 6, 7, 8. differences
        if grad_norm is not None and prev_grad_norm is not None:
            grad_diff = grad_norm - prev_grad_norm
            norm = torch.linalg.vector_norm(grad_diff)
            if norm > 0: directions.append(grad_diff / norm)

        if grad_norm is not None and velocity_norm is not None:
            grad_velocity_diff = grad_norm - velocity_norm
            norm = torch.linalg.vector_norm(grad_velocity_diff)
            if norm > 0: directions.append(grad_velocity_diff / norm)

        if prev_grad_norm is not None and prev_update_norm is not None:
            grad_update_diff = prev_grad_norm - prev_update_norm
            norm = torch.linalg.vector_norm(grad_update_diff)
            if norm > 0: directions.append(grad_update_diff / norm)

        if num_directions < len(directions):
            directions = random.choices(directions, k = num_directions) # this is not efficient but for now it will be it

        # 9+. Additional orthogonal directions if needed
        while len(directions) < num_directions:
            rand_dir = torch.randn_like(grad)
            for d in directions:
                rand_dir -= torch.dot(rand_dir, d) * d
            rand_dir /= rand_dir.norm() if rand_dir.norm() > 1e-8 else 1.0
            directions.append(rand_dir)

        return directions

    @torch.no_grad
    def estimate_hessian_global(self, pvec:torch.Tensor, directions:list[torch.Tensor], epsilon, closure, params: tz.tl.TensorList):
        hessian_estimates = []
        original_p = pvec.detach().clone()
        shape = pvec.shape
        for v in directions:
            # Perturb parameter in direction v
            params.from_vec_((original_p.view(-1) + epsilon * v).view(shape))
            self.zero_grad()
            with torch.enable_grad(): closure()
            grad_plus = params.grad.to_vec()

            params.from_vec_((original_p.view(-1) - epsilon * v).view(shape))
            self.zero_grad()
            with torch.enable_grad(): closure()
            grad_minus = params.grad.to_vec()

            pvec.copy_(original_p)
            # Finite difference Hessian-vector product
            Hv = (grad_plus - grad_minus) / (2 * epsilon)
            hessian_estimates.append(Hv)
        return hessian_estimates

    @torch.no_grad
    def estimate_hessian_paramwise(self, p:torch.Tensor, directions:list[torch.Tensor], epsilon, closure):
        hessian_estimates = []
        original_p = p.detach().clone()
        shape = p.shape
        for v in directions:
            # Perturb parameter in direction v
            p.copy_(original_p.view(-1) + epsilon * v).view(shape)
            self.zero_grad()
            with torch.enable_grad(): closure()
            assert p.grad is not None
            grad_plus = p.grad.detach().flatten()

            p.copy_(original_p.view(-1) - epsilon * v).view(shape)
            self.zero_grad()
            with torch.enable_grad(): closure()
            grad_minus = p.grad.detach().flatten()

            p.copy_(original_p)
            # Finite difference Hessian-vector product
            Hv = (grad_plus - grad_minus) / (2 * epsilon)
            hessian_estimates.append(Hv)
        return hessian_estimates

    @torch.no_grad
    def compute_update(self, grad, directions, hessian_estimates):
        update = torch.zeros_like(grad)
        for v, Hv in zip(directions, hessian_estimates):
            denominator = torch.dot(Hv, v)
            if abs(denominator) > 1e-8:
                alpha = torch.dot(grad, v) / denominator
                update += alpha * v
        return update

