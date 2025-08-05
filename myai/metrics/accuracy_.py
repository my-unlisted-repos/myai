import torch
from ._utils import _ensure_onehot

def accuracy(input:torch.Tensor, target:torch.Tensor, ):
    """
    Computes accuracy. Don't forget to convert the predictions to binary one-hot format!

    Args:
        input (torch.Tensor):
            `(B, C, *)` predicted labels, where C is the number of classes
        target (torch.Tensor):
            `(B, *)` or one-hot `(B, C, *)`

    Returns:
        torch.Tensor: Accuracy.
    """

    # argmax target if not argmaxed (input is always one hot encoded)
    if input.ndim == target.ndim:
        target = target.argmax(1)

    # Compute the fraction of correct predictions
    return torch.mean((input.argmax(1) == target).float())

def binary_accuracy(input:torch.Tensor, target:torch.Tensor, ):
    """
    Computes binary accuracy. Don' forget to convert predictions to binary format!

    Args:
        input (torch.Tensor):
            `(B, *)` , where * is 0 or more dims.
        target (torch.Tensor):
            `(B, *)` , where * is 0 or more dims.

    Returns:
        torch.Tensor: Accuracy.
    """
    input = input.squeeze()
    target = target.squeeze()
    return (input == target).float().mean()


def per_class_accuracy(input:torch.Tensor, target:torch.Tensor,):
    """
    Intersection over union metric often used for segmentation, also known as Jaccard index.
    Don't forget to convert the predictions to one-hot format!

    Args:
        input: prediction in `(B, C, *)`.
        target: ground truth in `(B, C, *)` or `(B, *)`.

    Returns:
        vector of len C with iou per each channel
    """
    input, target = _ensure_onehot(input, target)

    spatial_dims = list(range(2, target.ndim))

    correct = input == target
    if len(spatial_dims) > 0:
        correct = correct.float().mean(spatial_dims) # mean along all but B and C dims.

    return correct.mean(0) # mean along batch dim but not channel dim

def balanced_accuracy(input: torch.Tensor, target):
    return per_class_accuracy(input, target).mean()