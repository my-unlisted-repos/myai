import typing as T
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Iterator, MutableMapping

import numpy as np
import torch

from .base_logger import BaseLogger


class DictLogger(UserDict, BaseLogger, ):
    def log(self, step: int, metric: str, value: T.Any):
        if metric not in self: self[metric] = {step: value}
        else: self[metric][step] = value