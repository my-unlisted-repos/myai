# pylint:disable=signature-differs, not-callable
"""DeepSeek"""
import itertools
import random
from collections.abc import Callable

import numpy as np
import torch
import torchzero as tz
from torch import nn


#region PatternSearch
class PatternSearch(tz.core.TensorListOptimizer):
    """
    tests all straight directions. Includes adaptive step size.

    Args:
        params (_type_): params
        lr (float, optional): initial step size. Defaults to 1.
        npos (float, optional): step size increase when good direction has been found. Defaults to 1.1.
        nneg (float, optional): step size decrease when no directions decreased loss. Defaults to 0.5.
    """
    def __init__(self, params, lr: float=1, npos = 1.2, nneg = 0.5):
        super().__init__(params, {})
        self.npos = npos
        self.nneg = nneg
        self.step_size = lr

    @torch.no_grad
    def step(self, closure):
        p = self.get_params().with_requires_grad()
        params = p.to_vec()

        def objective_fn(vec):
            p.from_vec_(vec)
            return closure(False)

        best_value = objective_fn(params)
        best_direction = None

        for i in range(params.numel()):
            for direction in [(i, 1), (i, -1)]:
                # Perturb parameters in the current direction
                perturbed_params = params.clone()
                perturbed_params[direction[0]] += direction[1] * self.step_size
                value = objective_fn(perturbed_params)

                if value < best_value:
                    best_value = value
                    best_direction = direction

        if best_direction is not None:
            # Update parameters in the best direction found
            params[best_direction[0]] += best_direction[1] * self.step_size
            p.from_vec_(params)
            self.step_size *= self.npos
        else:
            self.step_size *= self.nneg
            # No improvement found, stop optimization

        return best_value
#endregion

#region BruteHillClimbing
class BruteHillClimbing(tz.core.TensorListOptimizer):
    """Test all straight and all diagonal directions.

    Args:
        params (_type_): _description_
        lr (float, optional): initial step size. Defaults to 0.01.
        npos (float, optional): step size increase when good direction has been found. Defaults to 1.1.
        nneg (float, optional): step size decrease when no directions decreased loss. Defaults to 0.5.
        max_tree_size (_type_, optional):
            maximum directions to generate (i added this to avoid OOMs and freezing the device). Defaults to 1_000_000.
    """
    def __init__(self, params, lr=0.01, npos = 1.2, nneg = 0.5, max_tree_size=1_000_000, ):
        super().__init__(params, {})
        self.step_size = lr
        self.npos = npos
        self.nneg = nneg
        self.dim = self.get_params().total_numel()

        # Define behavior tree: includes individual, pairwise, and main diagonal perturbations
        self.max_tree_size = max_tree_size
        self.behavior_tree = self.build_behavior_tree()

    def build_behavior_tree(self):
        """
        Build an enhanced behavior tree with individual, pairwise, and main diagonal perturbations.
        """
        tree = []
        n_dels = 0

        # Individual parameter perturbations
        for i in range(self.dim):
            tree.append(((i,), 1))   # Perturb parameter i by +step_size
            tree.append(((i,), -1))  # Perturb parameter i by -step_size
            if len(tree) > self.max_tree_size:
                del tree[random.randrange(0, len(tree))]
                n_dels += 1
                if n_dels == self.max_tree_size: break

        n_dels = 0
        # Pairwise parameter perturbations
        for pair in itertools.combinations(range(self.dim), 2):
            tree.append((pair, 1))   # Perturb parameters in pair by +step_size
            tree.append((pair, -1))  # Perturb parameters in pair by -step_size
            if len(tree) > self.max_tree_size:
                del tree[random.randrange(0, len(tree))]
                n_dels += 1
                if n_dels == self.max_tree_size: break

        # Main diagonal perturbations
        tree.append((tuple(range(self.dim)), 1))   # Perturb all parameters by +step_size
        tree.append((tuple(range(self.dim)), -1))  # Perturb all parameters by -step_size

        return tree

    @torch.no_grad
    def step(self, closure):
        """
        Perform optimization using the zeroth order optimizer.

        :param objective_fn: Function to minimize, takes parameters and returns a scalar
        """
        # for iteration in range(self.max_iterations):
        p = self.get_params()
        params = p.to_vec()

        def objective_fn(vec):
            p.from_vec_(vec)
            return closure(False)

        best_value = objective_fn(params)
        best_direction = None

        for direction, sign in self.behavior_tree:
            # Perturb parameters in the current direction
            perturbed_params = params.clone()
            for idx in direction:
                perturbed_params[idx] += sign * self.step_size
            value = objective_fn(perturbed_params)

            if value < best_value:
                best_value = value
                best_direction = (direction, sign)

        if best_direction is not None:
            # Update parameters in the best direction found
            for idx in best_direction[0]:
                params[idx] += best_direction[1] * self.step_size
                p.from_vec_(params)

            self.step_size *= self.npos
            # print(f"Iteration {iteration+1}: New best value {best_value}")
        else:
            # No improvement found, stop optimization
            # print(f"Iteration {iteration+1}: No improvement. Stopping.")
            self.step_size *= self.nneg

        return best_value
#endregion