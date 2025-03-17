# pylint:disable=signature-differs, not-callable
"""DeepSeek"""
import itertools
import random
from collections.abc import Callable

import numpy as np
import torch
import torchzero as tz
from torch import nn


# Define the neural network to suggest descent directions
class DirectionSuggester(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * input_size),  # Output mean and log_std
        )

    def forward(self, x):
        output = self.fc(x)
        mean, log_std = output.chunk(2, dim=-1)
        return mean, log_std


# Zeroth order optimizer class
class RLOptimizer(tz.core.TensorListOptimizer):
    """A little RL model optimizes the function. Needs a lot of model_opt_cls tuning!

    Args:
        params (_type_): _description_
        lr (float, optional): learning rate. Defaults to 0.01.
        model (torch.nn.Module | None, optional):
            should accept num_variables sized vector, and return a 2 times longer vector.
            If none defaults to a simple linear model (but a better model is recommended). Defaults to None.
        model_opt_cls (Callable): RL model optimizer class. Defauts to `lambda p: torch.optim.AdamW(p, 1e-3)`
    """

    def __init__(
        self,
        params,
        lr=0.01,
        model: torch.nn.Module | None = None,
        model_opt_cls: Callable = lambda p: torch.optim.AdamW(p, 1e-3),
    ):
        super().__init__(params, {})
        if model is None: model = DirectionSuggester(self.get_params().total_numel())
        self.model = model.to(self.get_params()[0])
        self.lr = lr
        #self.model_optimizer = optim.Adam(model.parameters(), lr=model_lr)
        self.model_optimizer = model_opt_cls(model.parameters())

    @torch.no_grad
    def step(self, closure): # pylint:disable=signature-differs
        # Get current parameters
        p = self.get_params()
        current_params = p.to_vec()

        self.new_params = None

        with torch.enable_grad():
            self._f_after = float('inf')
            def model_closure(backward=True):
                # Get suggested mean and log_std from the model
                mean, log_std = self.model(current_params)

                # Sample direction d from N(mean, exp(log_std)^2)
                std = torch.exp(log_std)
                std = std.abs().nan_to_num(0,0,0)
                normal = torch.distributions.Normal(mean, std)
                d = normal.sample()

                # Compute log probability of d
                log_prob = normal.log_prob(d).sum(-1, keepdim=True)

                # Update parameters with suggested direction
                self.new_params = current_params - self.lr * d.squeeze()

                with torch.no_grad():
                    # Compute function values before and after update
                    f_before = closure(False)
                    p.from_vec_(self.new_params)
                    self._f_after = closure(False)

                # Compute improvement
                delta = f_before - self._f_after

                # Compute loss as -delta * log_prob
                loss = -delta * log_prob

                # Zero gradients and backpropagate
                if backward:
                    self.model_optimizer.zero_grad()
                    loss.backward()

                with torch.no_grad():
                    p.from_vec_(current_params)

                return loss

            self.model_optimizer.step(model_closure)

        assert self.new_params is not None
        p.from_vec_(self.new_params)
        return self._f_after
# endregion