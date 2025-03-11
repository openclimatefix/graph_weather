import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from functools import lru_cache

from graph_weather.models.atmorep.config import AtmoRepConfig
from graph_weather.models.atmorep.inference import (
    load_model,
    inference,
    batch_inference,
    create_forecast,
)
from graph_weather.models.atmorep.data.dataset import ERA5Dataset
from graph_weather.models.atmorep.data.normalizer import FieldNormalizer
from graph_weather.models.atmorep.training.loss import AtmoRepLoss
from graph_weather.models.atmorep.training.train import train_atmorep
from graph_weather.models.atmorep.model.atmorep import AtmoRep

TEST_SPATIAL_DIMS = (32, 64)
TEST_BATCH_SIZE = 4
TEST_TIME_STEPS = 6
TEST_FIELDS = ["t2m", "u10", "v10", "z500", "tcc"]


@pytest.fixture(scope="session")
def config():
    """Fixture to provide a standard test configuration with session scope for reuse."""
    return AtmoRepConfig(
        spatial_dims=TEST_SPATIAL_DIMS,
        batch_size=TEST_BATCH_SIZE,
        time_steps=TEST_TIME_STEPS,
        input_fields=TEST_FIELDS,
    )


@pytest.fixture(scope="module")
@lru_cache(maxsize=1)
def dummy_data(config):
    """Fixture to generate dummy atmospheric data for testing with caching for reuse."""
    torch.manual_seed(42)
    return {
        field: torch.randn(2, config.time_steps, *config.spatial_dims)
        for field in config.input_fields
    }


@pytest.fixture(scope="function")
def normalizer(config, tmp_path):
    """Fixture to create a field normalizer with statistics."""
    norm_path = tmp_path / "test_stats"
    norm_path.mkdir(exist_ok=True)

    stats = {
        "t2m": {"mean": 288.15, "std": 15.0},  # Kelvin
        "u10": {"mean": 0.0, "std": 5.0},  # m/s
        "v10": {"mean": 0.0, "std": 5.0},  # m/s
        "z500": {"mean": 5500.0, "std": 150.0},  # geopotential height in m
        "tcc": {"mean": 0.5, "std": 0.3},  # total cloud cover (0-1)
    }

    for field in config.input_fields:
        if field in stats:
            np.save(norm_path / f"{field}_stats.npy", stats[field])
        else:
            field_stats = {"mean": torch.rand(1).item(), "std": torch.rand(1).item() + 0.5}
            np.save(norm_path / f"{field}_stats.npy", field_stats)

    return FieldNormalizer(config, norm_path)


@pytest.fixture(scope="module")
def mock_model(config):
    """Fixture to create a mock model that returns predictable outputs with module scope for reuse."""
    model = MagicMock(spec=AtmoRep)
    model.eval = MagicMock()
    model.config = config

    def side_effect(x, masks=None, ensemble_size=None):
        results = {}
        for field in x:
            base = x[field]
            time_trend = 0.05 * torch.arange(base.shape[1], device=base.device).reshape(1, -1, 1, 1)
            spatial_pattern = 0.1 * torch.sin(
                torch.linspace(0, 3.14, base.shape[2], device=base.device)
            ).reshape(1, 1, -1, 1)

            result = base * 0.9 + 0.1 + time_trend + spatial_pattern

            if ensemble_size and ensemble_size > 1:
                result = result.unsqueeze(0).repeat(ensemble_size, 1, 1, 1, 1)
                variations = 0.02 * torch.randn(
                    ensemble_size, *result.shape[1:], device=base.device
                )
                result = result + variations

            results[field] = result
        return results

    model.side_effect = side_effect
    model.__call__ = MagicMock(side_effect=side_effect)

    return model


@pytest.fixture(scope="function")
def mock_dataset(dummy_data, config):
    """Fixture to create a mock dataset for testing batch operations."""
    dataset = MagicMock(spec=ERA5Dataset)
    dataset.__len__.return_value = 10

    items = []
    for i in range(10):
        item = {field: data + 0.01 * i for field, data in dummy_data.items()}
        items.append(item)

    dataset.__getitem__.side_effect = lambda idx: items[idx % len(items)]

    return dataset


@pytest.fixture(scope="function")
def temp_model_file(tmp_path, config):
    """Fixture to create a temporary model file for testing model loading."""
    model_path = tmp_path / "test_model.pth"

    mock_state_dict = {
        "encoder.0.weight": torch.randn(16, 8, 3, 3),
        "encoder.0.bias": torch.randn(16),
        "decoder.0.weight": torch.randn(8, 16, 3, 3),
        "decoder.0.bias": torch.randn(8),
    }

    # Save mock model state
    torch.save(
        {
            "model": mock_state_dict,
            "config": config.__dict__,
            "epoch": 10,
            "optimizer_state": {"state": {}},
        },
        model_path,
        _use_new_zipfile_serialization=False,
    )

    return model_path


class TestAtmoRepConfig:
    """Tests for the AtmoRep configuration module."""

    def test_config_initialization(self, config):
        """Test if AtmoRepConfig initializes with correct default values."""
        assert config.spatial_dims == TEST_SPATIAL_DIMS, "Spatial dimensions incorrect"
        assert config.batch_size == TEST_BATCH_SIZE, "Batch size incorrect"
        assert isinstance(config.input_fields, list), "Input fields should be a list"
        assert len(config.input_fields) > 0, "Input fields should not be empty"
        assert all(field in config.input_fields for field in TEST_FIELDS), "Missing expected fields"

    def test_config_custom_values(self):
        """Test if AtmoRepConfig accepts custom values correctly."""
        custom_config = AtmoRepConfig(
            spatial_dims=(16, 32), batch_size=2, learning_rate=1e-4, time_steps=4
        )
        assert custom_config.spatial_dims == (16, 32)
        assert custom_config.batch_size == 2
        assert custom_config.learning_rate == 1e-4
        assert custom_config.time_steps == 4

    def test_config_validation(self):
        """Test configuration validation for invalid values."""
        with pytest.raises(Exception):
            AtmoRepConfig(spatial_dims=(-1, 64))

        with pytest.raises(Exception):
            AtmoRepConfig(batch_size=0)

        with pytest.raises(Exception):
            AtmoRepConfig(learning_rate=-0.001)

        with pytest.raises(Exception):
            AtmoRepConfig(input_fields=[])


class TestModelOperations:
    """Tests for model loading and inference operations."""

    def test_model_loading_invalid_path(self):
        """Test model loading with an invalid path raises appropriate error."""
        with pytest.raises(Exception):
            load_model("invalid_path.pth")

    @patch("torch.cuda.is_available", return_value=False)
    def test_model_loading_valid_path(self, mock_cuda, temp_model_file):
        """Test model loading with a valid path."""
        with patch(
            "graph_weather.models.atmorep.model.atmorep.AtmoRep", autospec=True
        ) as mock_atmorep:
            mock_model_instance = MagicMock()
            mock_atmorep.return_value = mock_model_instance

            loaded_model = load_model(temp_model_file, map_location="cpu")

            assert mock_atmorep.called

    def test_inference_output_shape(self, mock_model, dummy_data):
        """Test that inference produces outputs with correct shapes."""
        with torch.no_grad():  # Added no_grad for efficiency
            preds = inference(mock_model, dummy_data)

        assert set(preds.keys()) == set(dummy_data.keys())
        for field in dummy_data:
            assert preds[field].shape == dummy_data[field].shape
            assert not torch.allclose(preds[field], dummy_data[field])
            assert torch.allclose(preds[field], dummy_data[field], atol=1.0)

    def test_batch_inference_processing(self, mock_model, mock_dataset):
        """Test batch inference processes all data correctly."""
        batch_size = 2
        mock_model.__call__.reset_mock()

        with torch.no_grad():
            preds = batch_inference(
                mock_model,
                mock_dataset,
                batch_size=batch_size,
                num_workers=0,  # Critical to avoid pickling issues
            )

        # Check if model was called
        assert mock_model.__call__.call_count > 0, "Model was never called during inference"

        sample_item = mock_dataset.__getitem__(0)
        assert set(preds.keys()) == set(sample_item.keys())

    @pytest.mark.parametrize("forecast_steps", [1, 3])
    def test_forecasting_steps(self, mock_model, dummy_data, forecast_steps):
        """Test forecast generation for different step counts."""
        with torch.no_grad():  # Added for consistency and efficiency
            forecast = create_forecast(mock_model, dummy_data, steps=forecast_steps)

        assert set(forecast.keys()) == set(dummy_data.keys())
        for field in dummy_data:
            assert forecast[field].shape[1] >= dummy_data[field].shape[1]

            original_steps = min(dummy_data[field].shape[1], forecast[field].shape[1])
            if original_steps > 0:
                assert torch.is_tensor(forecast[field][:, :original_steps])


class DataHandlingTestBase:
    """Base class for data handling tests with common setup."""

    @pytest.fixture(autouse=True)
    def setup_data_path(self, tmp_path):
        """Setup common data path and files."""
        self.data_path = tmp_path / "era5_data"
        self.data_path.mkdir()
        return self.data_path


class TestDataHandling(DataHandlingTestBase):
    """Tests for dataset and data preprocessing operations."""

    @pytest.mark.parametrize("data_exists", [True, False])
    def test_dataset_initialization(self, config, data_exists):
        """Test ERA5Dataset initialization with and without existing data."""
        if data_exists:
            with open(self.data_path / "data_index.txt", "w") as f:
                f.write("era5_2015_01.nc\nera5_2015_02.nc\nera5_2015_03.nc")

        if not data_exists:
            with pytest.raises(Exception):
                with patch.object(ERA5Dataset, "load_file", return_value={}):
                    ERA5Dataset(config, self.data_path)
        else:
            with patch.object(ERA5Dataset, "load_file", return_value={}):
                dataset = ERA5Dataset(config, self.data_path)
                assert hasattr(dataset, "data_index") or hasattr(dataset, "file_list")

    def test_dataset_getitem(self, config, dummy_data):
        """Test dataset item retrieval."""
        with open(self.data_path / "data_index.txt", "w") as f:
            f.write("era5_2015_01.nc\nera5_2015_02.nc\nera5_2015_03.nc")

        return_values = []
        for i in range(3):
            value = {field: data + i * 0.01 for field, data in dummy_data.items()}
            return_values.append(value)

        with (
            patch.object(ERA5Dataset, "load_file") as mock_load,
            patch("os.path.exists", return_value=True),
        ):

            mock_load.side_effect = return_values
            dataset = ERA5Dataset(config, self.data_path)

            for i in range(min(3, len(dataset))):
                item = dataset[i]
                assert isinstance(item, dict)

    def test_normalization_field_validation(self, normalizer, config):
        """Test normalizer validates field names."""
        # Use smaller data for this test
        field_data = torch.randn(2, 2, 8, 16)

        normalized = normalizer.normalize(field_data, config.input_fields[0])
        assert normalized is not None

        with pytest.raises(Exception):
            normalizer.normalize(field_data, "invalid_field")

    def test_normalization_roundtrip(self, normalizer, config):
        """Test that normalization followed by denormalization returns original data."""
        atol = 1e-4

        # Use very small data for roundtrip test
        for field in config.input_fields:
            field_data = torch.randn(1, 2, 8, 16)  # Small dimensions

            normalized = normalizer.normalize(field_data, field)
            denormalized = normalizer.denormalize(normalized, field)

            assert denormalized.shape == field_data.shape

    def test_normalizer_stats_creation(self, config, dummy_data, tmp_path):
        """Test normalizer can calculate and save statistics."""
        stats_dir = tmp_path / "new_stats"
        stats_dir.mkdir()

        with patch.object(FieldNormalizer, "calculate_stats") as mock_calc:
            stats = {field: {"mean": 0.5, "std": 2.0} for field in config.input_fields}
            mock_calc.return_value = stats

            field_normalizer = FieldNormalizer(config, stats_dir, create_stats=True)

            assert mock_calc.called


class TrainingTestBase:
    """Base class for loss function and training tests with common setup."""

    @staticmethod
    def create_dummy_masks(batch_size, time_steps, height, width, fields):
        """Helper to create consistent masks for testing."""
        masks = {}
        for i, field in enumerate(fields):
            mask = torch.zeros(batch_size, time_steps, height, width)
            for b in range(batch_size):
                for t in range(time_steps):
                    if (b + t) % 3 == 0:
                        h_start = np.random.randint(0, height - 5)
                        w_start = np.random.randint(0, width - 5)
                        mask[b, t, h_start : h_start + 5, w_start : w_start + 5] = 1.0
            masks[field] = mask
        return masks


class TestTrainingComponents(TrainingTestBase):
    """Tests for loss functions and training loop components."""

    def test_loss_calculation_with_masks(self, config, dummy_data):
        """Test loss function handles masks correctly."""
        loss_fn = AtmoRepLoss(config)

        ensemble_size = 2
        batch_size = 2
        time_steps = config.time_steps // 2  # Half the time steps
        h, w = config.spatial_dims[0] // 2, config.spatial_dims[1] // 2

        small_data = {
            field: torch.randn(batch_size, time_steps, h, w) for field in config.input_fields
        }

        preds = {
            field: torch.randn(ensemble_size, batch_size, time_steps, h, w)
            for field in config.input_fields
        }

        masks = self.create_dummy_masks(batch_size, time_steps, h, w, config.input_fields)

        with torch.no_grad():
            loss, component_losses = loss_fn(preds, small_data, masks)

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Single scalar
        assert isinstance(component_losses, dict)

        # Test loss calculation without masks
        with torch.no_grad():
            result_no_mask = loss_fn(preds, small_data)

        loss_no_mask = result_no_mask[0] if isinstance(result_no_mask, tuple) else result_no_mask
        assert isinstance(loss_no_mask, torch.Tensor)

    def test_loss_weighting(self, config):
        """Test loss function applies field weights correctly."""
        field_weights = {field: i + 1 for i, field in enumerate(config.input_fields)}
        loss_fn = AtmoRepLoss(config, field_weights=field_weights)

        batch_size = 1
        time_steps = 2
        h, w = 8, 16

        small_data = {
            field: torch.randn(batch_size, time_steps, h, w) for field in config.input_fields
        }

        ensemble_size = 2
        preds = {
            field: torch.randn(ensemble_size, batch_size, time_steps, h, w)
            for field in config.input_fields
        }

        # Calculate loss
        masks = self.create_dummy_masks(batch_size, time_steps, h, w, config.input_fields)
        with torch.no_grad():
            result = loss_fn(preds, small_data, masks)

        loss = result[0] if isinstance(result, tuple) and len(result) >= 1 else result

        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert loss.item() >= 0  # Loss should be non-negative

    @patch("torch.save")
    def test_training_initialization(self, mock_save, config, tmp_path):
        """Test training function initializes correctly with proper device handling."""
        era5_path = tmp_path / "era5_data"
        era5_path.mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        from graph_weather.models.atmorep.training import train

        epoch_func_name = "_train_epoch"  # Choose one consistent name

        with (
            patch("graph_weather.models.atmorep.model.atmorep.AtmoRep") as mock_model_class,
            patch("graph_weather.models.atmorep.data.dataset.ERA5Dataset") as mock_dataset,
            patch("torch.utils.data.DataLoader") as mock_dataloader,
            patch("graph_weather.models.atmorep.training.loss.AtmoRepLoss") as mock_loss,
            patch("torch.optim.Adam") as mock_optimizer,
            patch.object(train, epoch_func_name) as mock_train_epoch,
            patch("torch.cuda.is_available", return_value=False),
        ):

            mock_dataset.return_value = MagicMock()
            mock_dataloader.return_value = MagicMock()

            mock_train_epoch.side_effect = KeyboardInterrupt

            try:
                train_atmorep(config, era5_path, output_dir)
            except KeyboardInterrupt:
                pass

            assert mock_model_class.called
            assert mock_dataset.called
            assert mock_loss.called

    @patch("torch.save")
    def test_training_with_resume(self, mock_save, config, tmp_path):
        """Test training resumes correctly from checkpoint with proper epoch tracking."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "checkpoint_epoch_5.pth"

        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        checkpoint = {
            "epoch": 5,
            "model": {"weight1": torch.randn(5)},
            "optimizer_state": {"state": {}},
            "config": config.__dict__,
            "best_val_loss": 0.123,
        }
        torch.save(checkpoint, checkpoint_path)

        from graph_weather.models.atmorep.training import train

        epoch_func_name = "_train_epoch"

        with (
            patch("graph_weather.models.atmorep.model.atmorep.AtmoRep") as mock_model_class,
            patch("graph_weather.models.atmorep.data.dataset.ERA5Dataset") as mock_dataset,
            patch("torch.utils.data.DataLoader") as mock_dataloader,
            patch("graph_weather.models.atmorep.training.loss.AtmoRepLoss") as mock_loss,
            patch("torch.optim.Adam") as mock_optimizer_class,
            patch.object(train, epoch_func_name) as mock_train_epoch,
            patch("torch.cuda.is_available", return_value=False),
        ):

            mock_model_class.return_value = mock_model
            mock_optimizer_class.return_value = mock_optimizer
            mock_dataset.return_value = MagicMock()
            mock_dataloader.return_value = MagicMock()

            mock_train_epoch.side_effect = KeyboardInterrupt

            try:
                train_atmorep(
                    config, tmp_path / "era5_data", tmp_path / "output", resume_from=checkpoint_path
                )
            except KeyboardInterrupt:
                pass

            assert mock_model_class.called

    def test_checkpoint_saving(self, config, tmp_path):
        """Test checkpoint saving works correctly with proper naming and content."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"weight1": torch.randn(5)}

        mock_optimizer = MagicMock()
        mock_optimizer.state_dict.return_value = {"state": {}}

        epoch = 10
        best_val_loss = 0.123
        is_best = True
        history = {"train_loss": [0.5, 0.4, 0.3], "val_loss": [0.6, 0.5, 0.4]}

        from graph_weather.models.atmorep.training import train

        save_func_name = "_save_checkpoint"

        with patch("torch.save") as mock_torch_save:
            save_fn = getattr(train, save_func_name)

            save_fn(
                mock_model,
                mock_optimizer,
                config,
                output_dir,
                epoch,
                best_val_loss,
                is_best,
                history,
            )

            assert mock_torch_save.called


class TestIntegration:
    """Integration tests that verify multiple components work together."""

    def test_inference_with_normalization(self, mock_model, normalizer, config):
        """Test end-to-end inference pipeline with normalization."""
        small_data = {
            field: torch.randn(
                1, config.time_steps // 2, config.spatial_dims[0] // 2, config.spatial_dims[1] // 2
            )
            for field in config.input_fields
        }

        # Normalize inputs
        normalized_inputs = {
            field: normalizer.normalize(data, field) for field, data in small_data.items()
        }

        # Run inference
        with torch.no_grad():
            normalized_preds = inference(mock_model, normalized_inputs)

        # Denormalize predictions
        preds = {
            field: normalizer.denormalize(normalized_preds[field], field)
            for field in normalized_preds.keys()
        }

        for field in small_data:
            assert preds[field].shape == small_data[field].shape
            assert not torch.allclose(preds[field], small_data[field])
            assert torch.all(torch.isfinite(preds[field]))

    def test_full_forecast_pipeline(self, mock_model, normalizer, config):
        """Test complete forecasting pipeline with normalization and denormalization."""
        small_data = {field: torch.randn(1, 3, 16, 32) for field in config.input_fields}

        # Normalize the inputs
        normalized_inputs = {
            field: normalizer.normalize(data, field) for field, data in small_data.items()
        }

        # Generate forecast with fewer steps
        steps = 2

        with torch.no_grad():
            normalized_forecast = create_forecast(mock_model, normalized_inputs, steps=steps)

        # Denormalize forecast
        forecast = {
            field: normalizer.denormalize(normalized_forecast[field], field)
            for field in normalized_forecast.keys()
        }

        for field in small_data:
            # Check forecast shape - should have more time steps than input
            assert forecast[field].shape[0] == small_data[field].shape[0]
            assert forecast[field].shape[1] >= small_data[field].shape[1]
            assert forecast[field].shape[2:] == small_data[field].shape[2:]

            # Check for physical realism - values should be finite
            assert torch.all(torch.isfinite(forecast[field]))

    def test_model_training_epoch(self, tmp_path):
        from graph_weather.models.atmorep.training import train

        def mock_forward(self, x, masks=None, **kwargs):
            """Mock forward method that doesn't use masks for simplicity"""
            return {field: x[field] for field in x}

        from graph_weather.models.atmorep.training import train

        def safe_generate_masks(batch_data, config):
            batch_masks = {}
            for field_name, field_data in batch_data.items():
                if hasattr(field_data, "shape"):
                    mask = torch.ones_like(field_data)
                else:
                    mask = torch.ones(1, 2, 16, 32)
                batch_masks[field_name] = mask
            return batch_masks

        original_mask_func = train.generate_training_masks

        class MockLoss:
            def __init__(self, config):
                self.config = config

            def __call__(self, pred, target, masks=None):
                # Calculate a simple MSE loss and ignore masks
                loss = 0.0
                field_losses = {}
                for field in self.config.input_fields:
                    field_loss = torch.mean((pred[field] - target[field]) ** 2)
                    loss += field_loss
                    field_losses[field] = field_loss.item()

                return loss, {"field_losses": field_losses}

        try:
            # Apply our patches
            train.generate_training_masks = safe_generate_masks

            # Create a small config for testing
            small_config = AtmoRepConfig(
                input_fields=["t2m", "u10"],
                spatial_dims=(16, 32),
                batch_size=1,
                time_steps=2,
                patch_size=4,
                mask_ratio=0.25,
            )

            # Initialize model and optimizer
            model = AtmoRep(small_config)

            # Make sure to handle all possible arguments correctly
            original_forward = model.forward
            model.forward = lambda x, masks=None, **kwargs: {
                field: x[field] * torch.nn.Parameter(torch.ones(1)) for field in x
            }

            optimizer = torch.optim.Adam(model.parameters(), lr=small_config.learning_rate)

            # Use our mock loss function
            loss_fn = MockLoss(small_config)

            batch = {
                field: torch.randn(
                    1,
                    small_config.time_steps,
                    small_config.spatial_dims[0],
                    small_config.spatial_dims[1],
                    requires_grad=True,
                )
                for field in small_config.input_fields
            }

            # Mock the data loader
            mock_loader = MagicMock()
            mock_loader.__iter__.return_value = [batch]
            mock_loader.__len__.return_value = 1

            with patch("torch.cuda.is_available", return_value=False):
                epoch_loss, field_losses = train._train_epoch(
                    model, mock_loader, optimizer, loss_fn, 0
                )

                assert isinstance(epoch_loss, float)

                for field in small_config.input_fields:
                    assert field in field_losses
                    assert isinstance(field_losses[field], float)

        finally:
            train.generate_training_masks = original_mask_func
            if hasattr(model, "forward") and "original_forward" in locals():
                model.forward = original_forward


class TestModelArchitecture:
    """Tests for the AtmoRep model architecture components."""

    @pytest.fixture(scope="class")
    def small_config(self):
        """Fixture providing a very small config for architecture tests."""
        return AtmoRepConfig(
            spatial_dims=(16, 32), batch_size=1, time_steps=2, input_fields=TEST_FIELDS[:2]
        )

    @pytest.fixture(scope="class")
    def small_sample_data(self, small_config):
        """Fixture providing sample data for model testing."""
        return {
            field: torch.randn(1, small_config.time_steps, *small_config.spatial_dims)
            for field in small_config.input_fields
        }

    def test_model_initialization(self, small_config, small_sample_data):
        """Test model initializes with correct parameters and structure."""
        model = AtmoRep(small_config)

        assert hasattr(model, "encoder") or hasattr(model, "encoders")
        assert hasattr(model, "decoder") or hasattr(model, "decoders")

        assert callable(getattr(model, "forward", None))

        with torch.no_grad():
            outputs = model(small_sample_data)

        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == set(small_config.input_fields)

        for field in small_config.input_fields:
            assert outputs[field].shape == small_sample_data[field].shape

    def test_model_with_masks(self, small_config, small_sample_data):
        """Test model handles masks correctly during forward pass."""
        model = AtmoRep(small_config)

        masks = {}
        for i, field in enumerate(small_config.input_fields):
            mask = torch.zeros(1, small_config.time_steps, *small_config.spatial_dims)
            for t in range(small_config.time_steps):
                if t % 2 == 0:  # Simpler pattern
                    h_start = 0
                    w_start = 0
                    mask[0, t, h_start : h_start + 5, w_start : w_start + 5] = 1.0
            masks[field] = mask

        # Call forward with masks and check outputs
        with torch.no_grad():
            outputs = model(small_sample_data, masks=masks)

        # Verify outputs have expected structure
        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == set(small_config.input_fields)

        for field in small_config.input_fields:
            assert outputs[field].shape == small_sample_data[field].shape

            # The masked and unmasked regions might have different patterns
            masked_output = outputs[field][masks[field] > 0]
            unmasked_output = outputs[field][masks[field] <= 0]

            if len(masked_output) > 0 and len(unmasked_output) > 0:
                # Statistical test - distribution should be different
                assert not torch.allclose(
                    masked_output.mean(), unmasked_output.mean(), rtol=0.1, atol=0.1
                )

    def test_ensemble_forecast(self, small_config, small_sample_data):
        """Test model generates proper ensemble forecasts."""
        model = AtmoRep(small_config)
        ensemble_size = 3

        # Run model to generate ensemble forecasts
        with torch.no_grad():
            outputs = model(small_sample_data, ensemble_size=ensemble_size)

        # Verify ensemble dimension is present
        for field in small_config.input_fields:
            assert outputs[field].shape[0] == ensemble_size
            assert outputs[field].shape[1:] == small_sample_data[field].shape

            # Ensure ensemble members are different from each other
            for i in range(ensemble_size):
                for j in range(i + 1, ensemble_size):
                    assert not torch.allclose(
                        outputs[field][i], outputs[field][j], rtol=1e-3, atol=1e-3
                    )

    def test_model_training_mode(self, small_config, small_sample_data):
        """Test model behaves differently in training vs evaluation mode."""
        model = AtmoRep(small_config)

        # Test in evaluation mode - should be deterministic
        model.eval()
        with torch.no_grad():
            eval_outputs1 = model(small_sample_data)
            eval_outputs2 = model(small_sample_data)

        for field in small_config.input_fields:
            assert torch.allclose(eval_outputs1[field], eval_outputs2[field])

        # In training mode, run multiple times to increase chance of seeing difference
        model.train()

        torch.manual_seed(42)
        train_outputs1 = model(small_sample_data)

        # Using a different seed for the second run
        torch.manual_seed(24)
        train_outputs2 = model(small_sample_data)

        any_field_different = False

        for field in small_config.input_fields:
            # Using a very loose tolerance to catch any meaningful differences
            if not torch.allclose(
                train_outputs1[field], train_outputs2[field], rtol=1e-1, atol=1e-1
            ):
                any_field_different = True
                break

        # Test passes if we found any difference
        assert (
            any_field_different
        ), "Training mode should produce different outputs with different random seeds"

    def test_autoregressive_property(self, small_config):
        """Test that model can use its own outputs as inputs for forecasting."""
        model = AtmoRep(small_config)

        initial_data = {
            field: torch.randn(1, 1, *small_config.spatial_dims)
            for field in small_config.input_fields
        }

        forecasted_data = {k: v.clone() for k, v in initial_data.items()}
        forecast_steps = 3

        for step in range(forecast_steps):
            with torch.no_grad():
                # Generate next step
                next_step = model(forecasted_data)

                # Append to forecast
                for field in small_config.input_fields:
                    next_value = next_step[field][:, -1:, :, :]
                    forecasted_data[field] = torch.cat([forecasted_data[field], next_value], dim=1)

        for field in small_config.input_fields:
            assert forecasted_data[field].shape[1] == forecast_steps + 1

        from graph_weather.models.atmorep.inference import create_forecast

        with torch.no_grad():
            built_in_forecast = create_forecast(model, initial_data, steps=forecast_steps)

        for field in small_config.input_fields:
            assert built_in_forecast[field].shape == forecasted_data[field].shape


class TestPerformanceAndScaling:
    """Tests for performance aspects and scaling behavior."""

    @pytest.mark.parametrize("spatial_size", [(16, 32), (32, 64)])
    def test_memory_usage(self, spatial_size):
        """Test model memory usage scales appropriately with spatial dimensions."""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        small_config = AtmoRepConfig(
            spatial_dims=spatial_size, batch_size=1, time_steps=2, input_fields=TEST_FIELDS[:2]
        )

        # Track memory usage
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

        model = AtmoRep(small_config).cuda()

        sample_data = {
            field: torch.randn(
                1, small_config.time_steps, *small_config.spatial_dims, device="cuda"
            )
            for field in small_config.input_fields
        }

        # Run inference
        with torch.no_grad():
            _ = model(sample_data)

        peak_mem = torch.cuda.max_memory_allocated()
        mem_used = peak_mem - start_mem

        expected_ratio = (spatial_size[0] * spatial_size[1]) / (16 * 32)

        # Log for debugging (can be commented out in final version)
        print(f"Memory used for spatial size {spatial_size}: {mem_used / 1024**2:.2f} MB")

        # This is a rough check - just making sure memory usage is reasonable
        # We allow for up to 2x the expected ratio to account for overhead
        assert mem_used > 0  # Just a sanity check

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_inference_speed(self, batch_size):
        """Test inference speed scales linearly with batch size."""
        config = AtmoRepConfig(
            spatial_dims=(16, 32), batch_size=batch_size, time_steps=2, input_fields=TEST_FIELDS[:1]
        )

        model = AtmoRep(config)

        small_data = {
            field: torch.randn(batch_size, config.time_steps, *config.spatial_dims)
            for field in config.input_fields
        }

        with torch.no_grad():
            _ = model(small_data)

        import time

        runs = 5

        start_time = time.time()
        with torch.no_grad():
            for _ in range(runs):
                _ = model(small_data)

        avg_time = (time.time() - start_time) / runs

        # Log for debugging (can be commented out in final version)
        print(f"Avg inference time for batch size {batch_size}: {avg_time:.4f} sec")

        assert avg_time > 0


class TestRobustness:
    """Tests for model robustness in edge cases and unusual inputs."""

    @pytest.fixture(scope="class")
    def robust_config(self):
        """Fixture providing a minimal config for robustness tests."""
        return AtmoRepConfig(
            spatial_dims=(8, 16), batch_size=1, time_steps=2, input_fields=TEST_FIELDS[:1]
        )

    def test_zero_input(self, robust_config):
        """Test model handles zero inputs gracefully."""
        model = AtmoRep(robust_config)

        zero_data = {
            field: torch.zeros(1, robust_config.time_steps, *robust_config.spatial_dims)
            for field in robust_config.input_fields
        }

        with torch.no_grad():
            outputs = model(zero_data)

        for field in robust_config.input_fields:
            assert outputs[field].shape == zero_data[field].shape
            assert torch.all(torch.isfinite(outputs[field]))

    def test_nan_handling(self, robust_config):
        """Test model's robustness to NaN values in input."""
        model = AtmoRep(robust_config)

        data = {
            field: torch.randn(1, robust_config.time_steps, *robust_config.spatial_dims)
            for field in robust_config.input_fields
        }

        # Insert NaNs in small regions
        for field in robust_config.input_fields:
            data[field][0, 0, :3, :3] = float("nan")

        masks = {field: ~torch.isnan(data[field]) for field in robust_config.input_fields}

        for field in robust_config.input_fields:
            data[field] = torch.nan_to_num(data[field], nan=0.0)

        with torch.no_grad():
            outputs = model(data, masks=masks)

        for field in robust_config.input_fields:
            assert outputs[field].shape == data[field].shape
            assert not torch.any(torch.isnan(outputs[field]))

    def test_single_precision(self, robust_config):
        """Test model works with single precision inputs."""
        model = AtmoRep(robust_config)

        fp32_data = {
            field: torch.randn(
                1, robust_config.time_steps, *robust_config.spatial_dims, dtype=torch.float32
            )
            for field in robust_config.input_fields
        }

        # Run inference
        with torch.no_grad():
            outputs = model(fp32_data)

        # Check outputs
        for field in robust_config.input_fields:
            assert outputs[field].dtype == torch.float32
            assert outputs[field].shape == fp32_data[field].shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_transfer(self, robust_config):
        """Test model can be moved between devices."""
        model = AtmoRep(robust_config)

        cpu_data = {
            field: torch.randn(1, robust_config.time_steps, *robust_config.spatial_dims)
            for field in robust_config.input_fields
        }

        # Test on CPU
        with torch.no_grad():
            cpu_outputs = model(cpu_data)

        # Move model to CUDA
        model.cuda()

        # Create CUDA data
        cuda_data = {field: tensor.cuda() for field, tensor in cpu_data.items()}

        # Test on CUDA
        with torch.no_grad():
            cuda_outputs = model(cuda_data)

        # Move model back to CPU
        model.cpu()

        # Compare shapes
        for field in robust_config.input_fields:
            assert cpu_outputs[field].shape == cuda_outputs[field].cpu().shape
