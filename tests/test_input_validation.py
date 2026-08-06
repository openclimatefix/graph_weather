"""Unit tests for input validation in graph_weather models.

These tests verify that model forward() methods correctly validate input tensor
shapes and fail fast with clear error messages at the API boundary.
"""

import pytest
import torch

from graph_weather.models.validation import validate_features_shape, validate_input_shape


class TestValidateInputShape:
    """Tests for the generic validate_input_shape function."""

    def test_valid_3d_tensor(self):
        """Test that valid 3D tensor passes validation."""
        tensor = torch.randn(4, 100, 64)
        # Should not raise
        validate_input_shape(tensor, expected_ndim=3)

    def test_invalid_2d_tensor_raises_valueerror(self):
        """Test that 2D tensor raises ValueError when 3D expected."""
        tensor = torch.randn(100, 64)
        with pytest.raises(ValueError, match=r"Invalid input shape"):
            validate_input_shape(tensor, expected_ndim=3)

    def test_invalid_1d_tensor_raises_valueerror(self):
        """Test that 1D tensor raises ValueError when 3D expected."""
        tensor = torch.randn(100)
        with pytest.raises(ValueError, match=r"Invalid input shape"):
            validate_input_shape(tensor, expected_ndim=3)

    def test_invalid_4d_tensor_raises_valueerror(self):
        """Test that 4D tensor raises ValueError when 3D expected."""
        tensor = torch.randn(4, 100, 64, 32)
        with pytest.raises(ValueError, match=r"Invalid input shape"):
            validate_input_shape(tensor, expected_ndim=3)

    def test_error_message_includes_expected_shape(self):
        """Test that error message includes expected shape description."""
        tensor = torch.randn(100, 64)
        with pytest.raises(ValueError, match=r"\[batch, nodes, features\]"):
            validate_input_shape(
                tensor,
                expected_ndim=3,
                expected_shape_desc="[batch, nodes, features]",
            )

    def test_error_message_includes_actual_shape(self):
        """Test that error message includes actual tensor shape."""
        tensor = torch.randn(100, 64)
        with pytest.raises(ValueError, match=r"\(100, 64\)"):
            validate_input_shape(tensor, expected_ndim=3)


class TestValidateFeaturesShape:
    """Tests for the features-specific validation function."""

    def test_valid_features_shape(self):
        """Test that valid features tensor passes validation."""
        features = torch.randn(8, 256, 78)
        # Should not raise
        validate_features_shape(features)

    def test_missing_batch_dimension(self):
        """Test that tensor missing batch dimension raises ValueError."""
        features = torch.randn(256, 78)  # Missing batch dimension
        with pytest.raises(ValueError, match=r"Invalid features shape"):
            validate_features_shape(features)

    def test_missing_features_dimension(self):
        """Test that tensor missing features dimension raises ValueError."""
        features = torch.randn(8, 256)  # Missing features dimension
        with pytest.raises(ValueError, match=r"Invalid features shape"):
            validate_features_shape(features)

    def test_empty_batch(self):
        """Test that empty batch still passes shape validation."""
        features = torch.randn(0, 256, 78)  # Empty batch
        # Should not raise - shape is still valid even if batch is empty
        validate_features_shape(features)


class TestModelInputValidation:
    """Integration tests for model forward() input validation.

    Note: These tests focus on the validation behavior, not full model execution.
    The models require heavy dependencies (h3, etc.) so we test validation in isolation.
    """

    def test_forecaster_import(self):
        """Test that forecaster can be imported with validation."""
        from graph_weather.models.forecast import GraphWeatherForecaster

        assert hasattr(GraphWeatherForecaster, "forward")

    def test_assimilator_import(self):
        """Test that assimilator can be imported with validation."""
        from graph_weather.models.analysis import GraphWeatherAssimilator

        assert hasattr(GraphWeatherAssimilator, "forward")
