import torch
from ._utils import _ensure_onehot

def dice(input:torch.Tensor, target:torch.Tensor, ):
    """
    Sørensen–Dice coefficient often used for segmentation. Defined as two intersections over sum. Equivalent to F1 score.

    input: prediction in `(B, C, *)`, C can be 1 for binary case.
    target: ground truth in `(B, C, *)` or `(B, *)`.
    reduce: `None`, `"mean"` or `"sum"`.

    returns: vector of len C with dice per each channel.
    """
    input, target = _ensure_onehot(input, target)

    spatial_dims = list(range(2, target.ndim))

    intersection = target.bool() & input.bool()
    sum = target + input

    if len(spatial_dims) > 0:
        intersection = intersection.sum(spatial_dims, dtype=torch.float32)
        sum = sum.sum(spatial_dims, dtype=torch.float32)

    return ((2*intersection) / sum).nanmean(0) # mean along batch dim but not channel dim