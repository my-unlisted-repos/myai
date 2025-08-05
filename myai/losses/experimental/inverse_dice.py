import typing as T
import warnings

import torch

from ...torch_tools import batched_one_hot_mask


class InverseDiceLoss(torch.nn.Module):
    def __init__(
        self,
        sigmoid=False,
        softmax=False,
        ignore_bg=False,
        eps: float | T.Literal["sum"] = 1e-5,
        jaccard = False,
        square_denominator = False,
        min_intersection: float = 1e-5,
        max_loss: float = 1,
        use_mse = False,
        mse_ord: int = 2,
        mse_multiplier:float = 1,
        logcosh = False,
        weight = None,
    ):
        """Loss based on the multiplicative inverse of the dice coefficient.

        :param sigmoid: Apply sigmoid to preds, defaults to False
        :param softmax: Apply softmax to preds, defaults to False
        :param ignore_bg: Whether to ignore first channel, defaults to False
        :param eps: Epsilon to avoid infinities. Can be `sum` where it is set to the sum of inputs and targets, defaults to 1e-5
        :param jaccard: Whether to use jaccard formula instead of dice, defaults to False
        :param batch: If True, calculates mean over entire batch, otherwise calculates mean for each sample separately, defaults to False
        :param square_denominator: Squares preds and targets in the denominator, defaults to False
        :param weight: A sequence of floats weights per each class, defaults to None
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
        self.min_intersection = min_intersection
        self.max_loss = max_loss
        self.use_mse = use_mse
        self.mse_multiplier = mse_multiplier
        self.mse_ord = mse_ord
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

        # values higher than zero to avoid division by 0
        inverse_mask = intersection > self.min_intersection

        # apply inverse dice formula to values where intersection is not 0
        loss = torch.zeros_like(sum)
        if self.jaccard:
            intersection_mask = intersection[inverse_mask]
            loss[inverse_mask] = ((sum[inverse_mask] - intersection_mask) / intersection_mask) - 1
        else:
            loss[inverse_mask] =  (sum[inverse_mask] / (2 * intersection[inverse_mask])) - 1

        # normalize loss to avoid large values when dividing by very small intersection
        large_loss_mask = loss > self.max_loss
        loss[large_loss_mask] = loss[large_loss_mask] / loss[large_loss_mask].detach()

        # calculate normal dice loss values with 0 intersection
        zero_mask = ~inverse_mask
        if self.use_mse:
            error = input - target
            if self.mse_ord % 2 != 0: error = error.abs()
            loss[zero_mask] = (error**self.mse_ord).sum(dim = spatial_dims)[zero_mask] * self.mse_multiplier
        else:
            # calculate the epsilon
            if self.eps == 'sum': eps = sum
            else: eps = self.eps
            # use normal dice formula
            loss[zero_mask] = 1 - ((2 * intersection[zero_mask] + eps) / (sum[zero_mask] + eps))

        if self.logcosh:
            loss = torch.log(torch.cosh(loss))

        # multiply loss by class weights, the loss is currently (B, C) so broadcastable into C
        if self.weight is not None:
            if loss.shape[-1] != len(self.weight):
                raise ValueError(f'Weights are {self.weight.shape} and dice is {loss.shape}')
            loss *= self.weight

        return loss.mean()
