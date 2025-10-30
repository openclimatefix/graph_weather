import pytest
import pandas as pd
import xarray as xr
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
from graph_weather.data.bufr_process import BUFR_processor


class TestBUFRProcessor:
    """Test BUFR_processor class."""

    @pytest.fixture
    def bufr_processor(self, mock_schema):
        return BUFR_processor(mock_schema)

    def test_processor_initialization(self, mock_schema):
        """Test processor initialization."""
        processor = BUFR_processor(mock_schema)
        assert processor.schema == mock_schema

    def test_processor_invalid_schema(self):
        """Test processor with invalid schema."""
        with pytest.raises(TypeError):
            BUFR_processor("invalid_schema")

    @patch("graph_weather.data.bufr_process.eccodes", create=True)
    def test_decode_bufr_file_success(self, mock_eccodes, bufr_processor, tmp_path):
        """Test successful BUFR file decoding."""
        test_file = tmp_path / "test.bufr"
        test_file.write_bytes(b"test bufr content")

        # Mock eccodes behavior
        mock_bufr_id = Mock()
        mock_eccodes.codes_bufr_new_from_file.side_effect = [mock_bufr_id, None]
        mock_iterator = Mock()
        mock_eccodes.codes_bufr_keys_iterator_new.return_value = mock_iterator
        mock_eccodes.codes_bufr_keys_iterator_next.side_effect = [True, False]
        mock_eccodes.codes_bufr_keys_iterator_get_name.return_value = "test_key"
        mock_eccodes.codes_get_string.return_value = "test_value"

        # Use the correct method name - decoder_bufr_files
        messages = bufr_processor.decoder_bufr_files(str(test_file))

        assert len(messages) == 1
        assert messages[0]["test_key"] == "test_value"

    def test_decode_bufr_file_not_found(self, bufr_processor, tmp_path):
        """Test BUFR file not found."""
        with pytest.raises(FileNotFoundError):
            bufr_processor.decoder_bufr_files(str(tmp_path / "nonexistent.bufr"))

    @patch.object(BUFR_processor, "decoder_bufr_files")
    def test_process_files_to_dataframe(self, mock_decode, bufr_processor, mock_schema):
        """Test processing BUFR file to DataFrame."""
        # Mock decoded messages
        mock_messages = [
            {"latitude": 45.0, "longitude": -120.0, "obsTime": "2023-01-01T12:00:00"},
            {"latitude": 46.0, "longitude": -121.0, "obsTime": "2023-01-01T12:30:00"},
        ]
        mock_decode.return_value = mock_messages

        # Mock schema mapping
        mock_schema.map_observation.side_effect = [
            {"OBS_TIMESTAMP": pd.Timestamp("2023-01-01T12:00:00"), "LAT": 45.0, "LON": -120.0},
            {"OBS_TIMESTAMP": pd.Timestamp("2023-01-01T12:30:00"), "LAT": 46.0, "LON": -121.0},
        ]

        with tempfile.NamedTemporaryFile(suffix=".bufr") as f:
            df = bufr_processor.process_files_to_dataframe(f.name)

        assert len(df) == 2
        assert "OBS_TIMESTAMP" in df.columns
        assert "LAT" in df.columns
        assert "LON" in df.columns
        assert df["LAT"].iloc[0] == 45.0

    @patch.object(BUFR_processor, "decoder_bufr_files")
    def test_process_files_to_dataframe_empty(self, mock_decode, bufr_processor):
        """Test processing BUFR file with no valid observations."""
        mock_decode.return_value = []  # No messages

        with tempfile.NamedTemporaryFile(suffix=".bufr") as f:
            df = bufr_processor.process_files_to_dataframe(f.name)

        assert df.empty

    @patch.object(BUFR_processor, "process_files_to_dataframe")
    def test_process_files_to_xarray(self, mock_process, bufr_processor):
        """Test processing BUFR file to xarray Dataset."""
        # Mock DataFrame
        mock_df = pd.DataFrame(
            {
                "OBS_TIMESTAMP": [
                    pd.Timestamp("2023-01-01T12:00:00"),
                    pd.Timestamp("2023-01-01T12:30:00"),
                ],
                "LAT": [45.0, 46.0],
                "LON": [-120.0, -121.0],
                "temperature": [20.0, 19.5],
                "pressure": [101325.0, 101300.0],
            }
        )
        mock_process.return_value = mock_df

        with tempfile.NamedTemporaryFile(suffix=".bufr") as f:
            ds = bufr_processor.process_files_to_xarray(f.name)

        assert "temperature" in ds.data_vars
        assert "pressure" in ds.data_vars
        assert "time" in ds.coords
        assert "lat" in ds.coords
        assert "lon" in ds.coords
        assert ds.attrs["source"] == "TEST"

    @patch.object(BUFR_processor, "process_files_to_dataframe")
    def test_process_files_to_parquet(self, mock_process, bufr_processor, tmp_path):
        """Test processing BUFR file to Parquet."""
        # Mock DataFrame
        mock_df = pd.DataFrame(
            {
                "OBS_TIMESTAMP": [pd.Timestamp("2023-01-01T12:00:00")],
                "LAT": [45.0],
                "LON": [-120.0],
                "temperature": [20.0],
            }
        )
        mock_process.return_value = mock_df

        output_file = tmp_path / "output.parquet"

        with tempfile.NamedTemporaryFile(suffix=".bufr") as f:
            bufr_processor.process_files_to_parquet(f.name, str(output_file))

        assert output_file.exists()
