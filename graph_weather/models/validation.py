"""Input validation utilities for graph_weather models.

Provides centralized validation functions to ensure input tensors have correct
shapes at the API boundary, enabling fail-fast behavior with clear error messages.
"""

import torch


def validate_input_shape(
    tensor: torch.Tensor,
    expected_ndim: int,
    name: str = "input",
    expected_shape_desc: str = "[batch, nodes, features]",
) -> None:
    """Validate that input tensor has the expected number of dimensions.

    Args:
        tensor: Input tensor to validate.
        expected_ndim: Expected number of dimensions.
        name: Name of the input parameter for error messages.
        expected_shape_desc: Human-readable description of expected shape.

    Raises:
        ValueError: If tensor does not have expected number of dimensions.
    """
    if tensor.ndim != expected_ndim:
        raise ValueError(
            f"Invalid {name} shape: expected {expected_ndim}D tensor with shape "
            f"{expected_shape_desc}, got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}"
        )


def validate_features_shape(features: torch.Tensor) -> None:
    """Validate that features tensor has shape [batch, nodes, features].

    Args:
        features: Input features tensor.

    Raises:
        ValueError: If features tensor is not 3D.
    """
    validate_input_shape(
        tensor=features,
        expected_ndim=3,
        name="features",
        expected_shape_desc="[batch, nodes, features]",
    )
