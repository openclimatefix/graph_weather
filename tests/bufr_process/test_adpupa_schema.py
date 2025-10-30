import pytest
import pandas as pd
from .test_common import MockADPUPASchema


class TestADPUPASchema:
    """Test ADPUPA schema implementation."""

    @pytest.fixture
    def adpupa_schema(self):
        return MockADPUPASchema()

    def test_schema_creation(self, adpupa_schema):
        """Test ADPUPA schema initialization."""
        assert adpupa_schema.source_name == "ADPUPA"
        assert len(adpupa_schema.field_mappings) > 0

    def test_required_mappings_present(self, adpupa_schema):
        """Test required NNJA coordinates are mapped."""
        output_names = {m.output_name for m in adpupa_schema.field_mappings.values()}
        assert "LAT" in output_names
        assert "LON" in output_names
        assert "OBS_TIMESTAMP" in output_names

    def test_map_observation(self, adpupa_schema, sample_adpupa_data):
        """Test observation mapping."""
        mapped = adpupa_schema.map_observation(sample_adpupa_data)

        assert mapped["LAT"] == 45.0
        assert mapped["LON"] == -120.0
        assert isinstance(mapped["OBS_TIMESTAMP"], pd.Timestamp)
        assert mapped["temperature"] == pytest.approx(26.85)  # 300K to C
        assert mapped["pressure"] == 101325.0
        assert mapped["dew_point"] == pytest.approx(16.85)  # 290K to C

    def test_map_observation_missing_fields(self, adpupa_schema):
        """Test mapping with missing fields."""
        test_message = {"latitude": 45.0, "longitude": -120.0, "obsTime": "2023-01-01T12:00:00"}

        mapped = adpupa_schema.map_observation(test_message)

        assert mapped["LAT"] == 45.0
        assert mapped["LON"] == -120.0
        assert mapped["temperature"] is None  # Missing field
