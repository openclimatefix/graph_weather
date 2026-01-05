import torch


def validate_model_input(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"Expected input to be torch.Tensor, got {type(x)}"
        )

    if x.ndim != 3:
        raise ValueError(
            f"Expected input shape [batch, nodes, features], got {x.shape}"
        )

    if x.size(1) <= 0 or x.size(2) <= 0:
        raise ValueError(
            f"Input tensor must have non-zero nodes and features, got {x.shape}"
        )
