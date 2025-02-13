import random
from abc import ABC, abstractmethod
from collections.abc import Callable

from ..rng import RNG


class Transform(ABC):
    _batched = False

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError(self.__class__.__name__ + " doesn't have `transform` method.")

    def batched_forward(self, x):
        raise NotImplementedError

    def reverse(self, x):
        raise NotImplementedError(self.__class__.__name__ + " doesn't have `reverse` method.")

    def batched(self, batched = True):
        """sets this transform to be batched (only if batched_forward is implemented)"""
        self._batched = batched
        return self

    def __call__(self, x):
        if self._batched: return self.batched_forward(x)
        return self.forward(x)



class RandomTransform(Transform, ABC):
    """note that transform applies to entire batch."""
    p:float

    def __init__(self, seed: int | None = None):
        self.rng = RNG(seed)

    def batched_forward(self, x):
        """this must implement p"""
        raise NotImplementedError


    def __call__(self, x):
        if self._batched: return self.batched_forward(x)
        if self.rng.random.random() < self.p: return self.forward(x)
        return x
