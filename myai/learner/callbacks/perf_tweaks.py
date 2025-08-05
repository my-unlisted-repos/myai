from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from ...torch_tools import performance_tweaks
from ...event_model import Callback

if TYPE_CHECKING:
    from ..learner import Learner

class PerformanceTweaks(Callback):
    order = -1000

    def __init__(
        self,
        cudnn_bench,
        onednn_fusion=True,
        detect_anomaly=False,
        checknan=False,
        autograd_profiler=False,
        emit_nvtx=False,
        deterministic=False,
        float32_matmul_precision = 'high',
        opt_einsum = True,
        opt_einsum_strategy = 'auto-hq',
        gradcheck=False,
        gradgradcheck=False,

    ):
        super().__init__()
        kwargs = locals().copy()
        del kwargs['self'], kwargs['__class__']
        self.kwargs = kwargs

    def enter(self, learner: "Learner"):
        performance_tweaks(**self.kwargs)
