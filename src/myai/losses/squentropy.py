import typing as T
import warnings

import torch

from ..torch_tools import batched_one_hot_mask

class Squentropy(torch.nn.Module):
    """Hui, L., Belkin, M., & Wright, S. (2023, July). Cut your losses with squentropy.
    In International Conference on Machine Learning (pp. 14114-14131). PMLR.

    Cross-entropy loss plus average square loss over the incorrect classes.


    Args:
        true_scale (float, optional): rescales loss at true label. Defaults to 1.
        onehot_scale (float, optional): rescales the one-hot encoding. Defaults to 1.
        ignore_bg (bool, optional): whether to ignore channel under index 0, which is commonly background in segmentation tasks. Defaults to False.
        weight (_type_, optional): weights of ecah class. Defaults to None.
        ord (int, optional): order of the loss (1 for L1 loss, 2 for MSE, etc). Defaults to 2.

    Signature:
        input: `(B, C, *)`. Unnormalized logits without softmax or sigmoid.
        target: `(B, C, *)` or `(B, *)`
    """
    def __init__(
        self,
        true_scale: float = 1,
        onehot_scale: float = 1,
        ignore_bg=False,
        weight = None,
        ord = 2,
    ):
        super().__init__()
        self.ignore_bg = ignore_bg
        if weight is not None and not isinstance(weight, torch.Tensor): weight = torch.as_tensor(weight, dtype=torch.float32)
        self.weight = weight
        self.true_scale = true_scale
        self.onehot_scale = onehot_scale
        self.ord = ord

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        input: `(B, C, *)`. Unnormalized logits without softmax or sigmoid.
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

        # check shapes
        if input.shape != target.shape:
            raise ValueError(f'{input.shape = } and {target.shape = }')

        # make sure target has correct dtype
        target = target.to(input.dtype, copy = False)

        # cross-entropy
        if num_classes == 1:
            ce_loss = -(target * torch.nn.functional.logsigmoid(input) + (1 - target) * torch.nn.functional.logsigmoid(1 - input)) # pylint:disable = not-callable
        else:
            ce_loss = -torch.nn.functional.log_softmax(input, dim = 1) * target

        # multiply one-hot vector by M
        if self.onehot_scale != 1: target = target * self.onehot_scale

        # square loss (or other order)
        square_loss = input - target
        if self.ord % 2 != 0: square_loss = square_loss.abs()
        if self.ord != 1: square_loss = square_loss**self.ord

        # rescale loss at true label by k
        if self.true_scale != 1:
            true_mask = target == 1
            square_loss[true_mask] = square_loss[true_mask] * self.true_scale

        # add the losses after averaging spatial dims
        # dims to sum over
        spatial_dims = list(range(2, input.ndim))
        if len(spatial_dims) > 0:
            square_loss = square_loss.mean(dim = spatial_dims)
            ce_loss = ce_loss.mean(dim = spatial_dims)

        loss = square_loss + ce_loss

        # multiply loss by class weights, the loss is currently (B, C) so broadcastable into C
        if self.weight is not None:
            self.weight = self.weight.to(loss.device, copy=False)
            if loss.shape[-1] != len(self.weight):
                raise ValueError(f'Weights are {self.weight.shape} and dice is {loss.shape}')
            loss *= self.weight

        return loss.mean()
