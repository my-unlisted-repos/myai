# pylint:disable=signature-differs, not-callable
"""DeepSeek"""
import itertools
import random
from collections.abc import Callable

import numpy as np
import torch
import torchzero as tz
from torch import nn



class CoordSearch(tz.core.TensorListOptimizer):
    """
    Something like adaptive coordinate wise line search, actually works well

    Parameters:
        params ( ): The parameters to optimize
        lr (float): Initial step size for line search, gets adapted based on successful/unsuccessful steps.
    """
    def __init__(self, params, lr=1.):
        super().__init__(params, {})
        self.delta = lr

    @torch.no_grad
    def step(self, closure):
        p = self.get_params().with_requires_grad()
        params = p.to_vec()

        n = params.numel()

        def func(vec):
            p.from_vec_(vec)
            return closure(False)

        f_old = func(params)
        f_new = None
        improvement = 0.0
        # Cycle through each search direction
        for i in range(n):
            d = torch.zeros_like(params)
            d[i] = 1

            # Evaluate function at current params and params + delta*d
            params_plus = params + self.delta * d
            f_plus = func(params_plus)
            # Determine the sign of improvement
            if f_plus < f_old:
                # Move in the positive direction
                while True:
                    params_new = params + self.delta * d
                    f_new = func(params_new)
                    if f_new < f_old:
                        params = params_new
                        f_old = f_new
                        self.delta *= 2  # Increase step size
                    else:
                        self.delta /= 2  # Decrease step size
                        break
                improvement += f_old - f_new
            else:
                params_minus = params - self.delta * d
                f_minus = func(params_minus)
                if f_minus < f_old:
                    # Move in the negative direction
                    while True:
                        params_new = params - self.delta * d
                        f_new = func(params_new)
                        if f_new < f_old:
                            params = params_new
                            f_old = f_new
                            self.delta *= 2  # Increase step size
                        else:
                            self.delta /= 2  # Decrease step size
                            break
                    improvement += f_old - f_new
                else:
                    # No improvement in this direction
                    self.delta /= 2

        p.from_vec_(params)
        return f_new if f_new is not None else f_old
