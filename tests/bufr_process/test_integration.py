import pytest
import tempfile
from graph_weather.data.bufr_process import BUFR_dataloader, DataSourceSchema, ADPUPA_schema, CRIS_schema, BUFR_processor


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_schema_registry_completeness(self):
        """Test that all schemas in registry can be instantiated."""
        for schema_name, schema_class in BUFR_dataloader.SCHEMA_REGISTRY.items():
            schema = schema_class()
            assert isinstance(schema, DataSourceSchema)
            assert schema.source_name == schema_name
    
    def test_end_to_end_mock_processing(self):
        """Test complete mock processing pipeline."""
        with tempfile.NamedTemporaryFile(suffix='_adpupa.bufr') as f:
            loader = BUFR_dataloader(f.name) 
            
            assert loader.schema_name == 'ADPUPA'
            assert isinstance(loader.schema, ADPUPA_schema)
            assert isinstance(loader.processor, BUFR_processor)
            assert loader.processor.schema == loader.schema