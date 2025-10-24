import pytest
import pandas as pd
from graph_weather.data.bufr_process import FieldMapping


class TestFieldMapping:
    """Test FieldMapping dataclass."""
    
    def test_field_mapping_creation(self):
        """Test FieldMapping initialization."""
        mapping = FieldMapping(
            source_name="temperature",
            output_name="temp",
            dtype=float,
            description="Temperature field"
        )
        
        assert mapping.source_name == "temperature"
        assert mapping.output_name == "temp"
        assert mapping.dtype == float
        assert mapping.description == "Temperature field"
        assert mapping.required is True
        assert mapping.transform_fn is None
    
    def test_field_mapping_apply_no_transform(self):
        """Test apply method without transformation function."""
        mapping = FieldMapping(
            source_name="pressure",
            output_name="pres",
            dtype=float
        )
        
        result = mapping.apply(1013.25)
        assert result == 1013.25
    
    def test_field_mapping_apply_with_transform(self):
        """Test apply method with transformation function."""
        mapping = FieldMapping(
            source_name="temp_k",
            output_name="temp_c",
            dtype=float,
            transform_fn=lambda x: x - 273.15
        )
        
        result = mapping.apply(300.0)
        assert result == pytest.approx(26.85)
    
    def test_field_mapping_apply_none_value(self):
        """Test apply method with None value."""
        mapping = FieldMapping(
            source_name="missing",
            output_name="missing_out",
            dtype=float
        )
        
        result = mapping.apply(None)
        assert result is None