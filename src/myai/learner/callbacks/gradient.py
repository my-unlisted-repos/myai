from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torchzero.core import OptimizerModule
from torchzero.modules import LaplacianSmoothing as TorchzeroLaplacianSmoothing, SetGrad
from torchzero.modules import (gradient_laplacian_smoothing_, normalize_grad_,
                               sign_grad_)
from torchzero.optim import Modular

from ...event_model import Callback

if TYPE_CHECKING:
    from ..learner import Learner


class GradClipNorm(Callback):
    """Gradient clipping by norm."""

    def __init__(self, max_norm: float, norm_type: float = 2,):
        super().__init__()
        self.max_norm = max_norm
        self.norm_type = norm_type

        self._learner_text = f'GradClipL{norm_type}Norm{max_norm}'

    def after_backward(self, learner: "Learner"):
        torch.nn.utils.clip_grad_norm_(learner.model.parameters(), self.max_norm, norm_type=self.norm_type)


class GradClipValue(Callback):
    """Gradient clipping by value."""

    def __init__(self, max_value: float):
        super().__init__()
        self.max_value = max_value

        self._learner_text = f'GradClip{max_value}'

    def after_backward(self, learner: "Learner"):
        torch.nn.utils.clip_grad_value_(learner.model.parameters(), self.max_value)

class GradNorm(Callback):
    """Gradient normalization."""
    def __init__(self, norm_value: float, norm_type: float = 2, min:float = 0):
        super().__init__()
        self.norm_value = norm_value
        self.norm_type = norm_type
        self.min = min

        self._learner_text = f'GradL{norm_type}Norm{norm_value}'

    def after_backward(self, learner: "Learner"):
        normalize_grad_(learner.model.parameters(), self.norm_value, min = self.min, ord = self.norm_type)


class GradSign(Callback):
    """Takes the sign of the gradient."""
    _learner_text = 'GradSign'

    def after_backward(self, learner: "Learner"):
        sign_grad_(learner.model.parameters())

class LaplacianSmoothing(Callback):
    """Applies laplacian smoothing to the gradient."""
    def __init__(self, sigma: float = 1, layerwise: bool = True):
        super().__init__()
        self.sigma = sigma
        self.layerwise = layerwise
        self.smoother = None

        self._learner_text = f'{"Layerwise" if layerwise else ""}LaplacianSmoothing{sigma}'

    def after_backward(self, learner: "Learner"):
        if self.smoother is None:
            # we create a smoother because it will cache the denominator which will be faster
            self.smoother = Modular(
                learner.model.parameters(),
                TorchzeroLaplacianSmoothing(sigma = self.sigma, layerwise=self.layerwise),
                SetGrad(),
            )

        self.smoother.step()

class TorchzeroModule(Callback):
    def __init__(self, modules: OptimizerModule | Iterable[OptimizerModule], set_grad=True):
        super().__init__()
        if isinstance(modules, OptimizerModule): modules = [modules]
        self.modules = list(modules)
        if set_grad: self.modules.append(SetGrad())
        self.opt = None

    def before_fit(self, learner: "Learner"):
        self.opt = Modular(learner.model.parameters(), self.modules)

    def after_backward(self, learner: "Learner"):
        if self.opt is None:
            self.opt = Modular(learner.model.parameters(), self.modules)

        closure = learner.make_closure(learner.batch)
        self.opt.step(closure)