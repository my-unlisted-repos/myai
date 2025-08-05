import torch

from ._utils import _ensure_onehot


def iou(input:torch.Tensor, target:torch.Tensor,):
    """
    Intersection over union metric often used for segmentation, also known as Jaccard index.

    input: prediction in `(B, C, *)`.
    target: ground truth in `(B, C, *)` or `(B, *)`.
    reduce: `None`, `"mean"` or `"sum"`.

    returns: vector of len C with iou per each channel
    """
    input, target = _ensure_onehot(input, target)
    input = input.bool()
    target = target.bool()

    spatial_dims = list(range(2, target.ndim))

    intersection = input & target
    union = input | target

    if len(spatial_dims) > 0:
        intersection = intersection.sum(spatial_dims, dtype=torch.float32)
        union = union.sum(spatial_dims, dtype=torch.float32)

    return (intersection / union).nanmean(0) # mean along batch dim but not channel dim