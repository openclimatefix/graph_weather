import pytest
import pandas as pd
from .test_common import MockCRISSchema


class TestCRISSchema:
    """Test CrIS schema implementation."""

    @pytest.fixture
    def cris_schema(self):
        return MockCRISSchema()

    def test_schema_creation(self, cris_schema):
        """Test CrIS schema initialization."""
        assert cris_schema.source_name == "CrIS"
        assert len(cris_schema.field_mappings) > 0

    def test_map_observation(self, cris_schema, sample_cris_data):
        """Test CrIS observation mapping."""
        mapped = cris_schema.map_observation(sample_cris_data)

        assert mapped["LAT"] == 30.0
        assert mapped["LON"] == -100.0
        assert isinstance(mapped["OBS_TIMESTAMP"], pd.Timestamp)
        assert mapped["temperature"] == pytest.approx(6.85)  # 280K to C
        assert mapped["pressure"] == 85000.0
        assert mapped["qc_flag"] == 1
