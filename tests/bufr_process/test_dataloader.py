import pytest
import pandas as pd
import xarray as xr
import tempfile
from pathlib import Path
from unittest.mock import patch
from graph_weather.data.bufr_process import BUFR_dataloader, ADPUPA_schema, BUFR_processor


class TestBUFRDataLoader:
    """Test BUFR_dataloader class."""

    def test_dataloader_initialization_with_schema(self):
        """Test dataloader initialization with explicit schema."""
        with tempfile.NamedTemporaryFile(suffix=".bufr") as f:
            loader = BUFR_dataloader(f.name, schema_name="ADPUPA")

        assert loader.schema_name == "ADPUPA"
        assert isinstance(loader.schema, ADPUPA_schema)
        assert isinstance(loader.processor, BUFR_processor)

    def test_dataloader_initialization_infer_schema(self):
        """Test dataloader initialization with schema inference."""
        test_cases = [
            ("test_adpupa.bufr", "ADPUPA"),
            ("test_CRIS_data.bufr", "CrIS"),
            ("unknown_file.bufr", "ADPUPA"),  # Default case
        ]

        for filename, expected_schema in test_cases:
            with tempfile.NamedTemporaryFile(suffix=filename) as f:
                loader = BUFR_dataloader(f.name)
                assert loader.schema_name == expected_schema

    def test_dataloader_initialization_invalid_schema(self):
        """Test dataloader initialization with invalid schema."""
        with tempfile.NamedTemporaryFile(suffix=".bufr") as f:
            with pytest.raises(ValueError, match='Unknown schema "INVALID"'):
                BUFR_dataloader(f.name, schema_name="INVALID")

    @patch.object(BUFR_processor, "process_files_to_dataframe")
    def test_to_dataframe(self, mock_process):
        """Test to_dataframe method."""
        mock_df = pd.DataFrame(
            {"OBS_TIMESTAMP": [pd.Timestamp("2023-01-01T12:00:00")], "LAT": [45.0], "LON": [-120.0]}
        )
        mock_process.return_value = mock_df

        with tempfile.NamedTemporaryFile(suffix=".bufr") as f:
            loader = BUFR_dataloader(f.name, schema_name="ADPUPA")
            df = loader.to_dataframe()

        assert len(df) == 1
        mock_process.assert_called_once_with(str(Path(f.name)))

    @patch.object(BUFR_processor, "process_files_to_xarray")
    def test_to_xarray(self, mock_process):
        """Test to_xarray method."""
        mock_ds = xr.Dataset(
            {"temperature": (["obs"], [20.0]), "pressure": (["obs"], [101325.0])},
            coords={
                "obs": [0],
                "time": ("obs", [pd.Timestamp("2023-01-01T12:00:00")]),
                "lat": ("obs", [45.0]),
                "lon": ("obs", [-120.0]),
            },
        )
        mock_process.return_value = mock_ds

        with tempfile.NamedTemporaryFile(suffix=".bufr") as f:
            loader = BUFR_dataloader(f.name, schema_name="ADPUPA")
            ds = loader.to_xarray()

        assert "temperature" in ds.data_vars
        mock_process.assert_called_once_with(str(Path(f.name)))

    @patch.object(BUFR_processor, "process_files_to_parquet")
    def test_to_parquet(self, mock_process, tmp_path):
        """Test to_parquet method."""
        output_file = tmp_path / "test_output.parquet"

        with tempfile.NamedTemporaryFile(suffix=".bufr") as f:
            loader = BUFR_dataloader(f.name, schema_name="ADPUPA")
            loader.to_parquet(str(output_file))

        mock_process.assert_called_once_with(str(Path(f.name)), str(output_file))

    @patch.object(BUFR_processor, "decoder_bufr_files")
    def test_iterator(self, mock_decode):
        """Test dataloader iterator."""
        mock_messages = [{"lat": 45.0, "lon": -120.0}, {"lat": 46.0, "lon": -121.0}]
        mock_decode.return_value = mock_messages

        with tempfile.NamedTemporaryFile(suffix=".bufr") as f:
            loader = BUFR_dataloader(f.name, schema_name="ADPUPA")

            # Mock the schema's map_observation to return simple data
            loader.schema.map_observation = lambda x: {"LAT": x["lat"], "LON": x["lon"]}

            observations = list(loader)

        assert len(observations) == 2
        assert observations[0]["LAT"] == 45.0
        assert observations[1]["LAT"] == 46.0
