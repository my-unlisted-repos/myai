import typing as T
import warnings

import torch

from ..torch_tools import batched_one_hot_mask


class DiceLoss(torch.nn.Module):
    def __init__(
        self,
        sigmoid=False,
        softmax=False,
        ignore_bg=False,
        eps: float | T.Literal["sum"] = 1e-5,
        jaccard = False,
        square_denominator = False,
        weight = None,
        logcosh = False,
    ):
        """Dice loss = 1 - ((2*intersections + eps) / (sums + eps))

        :param sigmoid: _description_, defaults to False
        :param softmax: _description_, defaults to False
        :param ignore_bg: _description_, defaults to False
        :param eps: _description_, defaults to 1e-5
        :param jaccard: _description_, defaults to False
        :param square_denominator: _description_, defaults to False
        :param weight: _description_, defaults to None
        :param logcosh: _description_, defaults to False
        """
        super().__init__()
        self.ignore_bg = ignore_bg
        self.eps = eps

        self.sigmoid = sigmoid
        self.softmax = softmax
        if weight is not None and not isinstance(weight, torch.Tensor): weight = torch.as_tensor(weight, dtype=torch.float32)
        self.weight = weight
        self.jaccard = jaccard
        self.square_denominator = square_denominator
        self.logcosh = logcosh

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        input: `(B, C, *)`
        target: `(B, C, *)` or `(B, *)`
        """
        num_classes = input.shape[1]

        # make sure target is one hot encoded
        if input.ndim - target.ndim == 1:

            # non-binary case
            if num_classes > 1:
                target = batched_one_hot_mask(target, num_classes = input.shape[1])

            # in binary case target is already 0s and 1s, we add the channel dimension to it.
            else:
                target = target.unsqueeze(1)

        # remove background channel
        if self.ignore_bg:
            input = input[:,1:]
            target = target[:, 1:]

        # apply activation
        if self.sigmoid:
            input = torch.sigmoid(input)
        elif self.softmax:
            if num_classes == 1: warnings.warn('Using softmax with binary segmentation')
            input = torch.softmax(input, dim = 1)

        # chck shapes
        if input.shape != target.shape:
            raise ValueError(f'{input.shape = } and {target.shape = }')

        # make sure target has correct dtype
        target = target.to(input.dtype, copy = False)

        # dims to sum over, either calculate over entire batch or each sample separately
        spatial_dims = list(range(2, input.ndim))
        has_spatial_dims = len(spatial_dims) > 0

        # we have two (B, C, *) tensors, now we compute CHANNEL WISE intersection and sum
        intersection = input * target
        if has_spatial_dims: intersection = intersection.sum(dim = spatial_dims)

        if self.square_denominator:
            sum = input**2 + target**2
        else:
            sum = input + target
        if has_spatial_dims: sum = sum.sum(dim = spatial_dims)

        # calculate the epsilon
        if self.eps == 'sum': eps = sum
        else: eps = self.eps

        # dice formula
        if self.jaccard:
            loss = 1 - ((intersection + eps) / ((sum - intersection) + eps))
        else:
            loss = (1 - ((2 * intersection + eps) / (sum + eps)))

        if self.logcosh:
            loss = torch.log(torch.cosh(loss))

        # multiply loss by class weights, the loss is currently (B, C) so broadcastable into C
        if self.weight is not None:
            self.weight = self.weight.to(loss.device, copy=False)
            if loss.shape[-1] != len(self.weight):
                raise ValueError(f'Weights are {self.weight.shape} and dice is {loss.shape}')
            loss *= self.weight

        loss.mean()
