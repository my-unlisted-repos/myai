import typing as T

import numpy as np

from ..python_tools import identity
from ..transforms import normalize, tonumpy


def _log10_norm(x): return np.log10(x)
def _log2_norm(x): return np.log2(x)
_NORMS = {
    None: normalize,
    "no": identity,
    "none": identity,
    "norm": normalize,
    "normalize": normalize,
    "log": _log10_norm,
    "log10": _log10_norm,
    "_log2_norm": _log2_norm,
}

# matplotlib ignores norm when input is RGB
# we fix that
# matplotlib.scale.get_scale_names()
# >> ['asinh', 'function', 'functionlog', 'linear', 'log', 'logit', 'symlog']
# no idea where it converts those to norms
def _normalize(value, norm: str | T.Callable | None):
    if isinstance(norm, str) or norm is None: norm_c = _NORMS[norm]
    else: norm_c = norm
    return norm_c(tonumpy(value))
