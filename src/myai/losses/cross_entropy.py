import torch


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Finally, after months of work, scientists have been able obtain the Cross Entropy Loss with Automatic Casting.

    Dtypes:
    If you don't want any casting to happend:
    If target is one hot encoded, it needs to have same dtype as input. Otherwise it needs to be int64."""
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,):
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != input.ndim:
            return super().forward(input, target.to(torch.int64, copy = False))
        return super().forward(input, target.to(input.dtype, copy = False))



class BCELoss(torch.nn.BCELoss):
    """Finally, after months of work, scientists have been able obtain the Binary Cross Entropy Loss
    with Automatic Casting."""
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        super().__init__(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target.to(input.dtype, copy = False))

class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """Finally, after months of work, scientists have been able obtain the Binary Cross Entropy with logits Loss
    with Automatic Casting."""
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,):
        super().__init__(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.ndim - target.ndim == 1: # (batch, H, W) -> (batch, 1, H, W)
            return super().forward(input, target.unsqueeze(1))
        return super().forward(input, target.to(input.dtype, copy = False))