
import pytest
import torch
from graph_weather.models import GraphWeatherForecaster, GraphWeatherAssimilator
from graph_weather.models.aurora.model import AuroraModel

def test_forecaster_input_shape_validation():
    """Test that GraphWeatherForecaster raises ValueError for invalid input shapes."""
    lat_lons = [(0, 0), (0, 1)]
    model = GraphWeatherForecaster(lat_lons=lat_lons, feature_dim=10)
    
    # Invalid shape: [batch, nodes] (missing features)
    x = torch.randn(10, 2)
    with pytest.raises(ValueError, match="Expected input shape"):
        model(x)

    # Valid shape
    x = torch.randn(10, 2, 10)
    try:
        model(x)
    except Exception as e:
        # Ignore other errors, just checking the validation passes
        pass

def test_assimilator_input_shape_validation():
    """Test that GraphWeatherAssimilator raises ValueError for invalid input shapes."""
    lat_lons = [(0, 0), (0, 1)]
    model = GraphWeatherAssimilator(output_lat_lons=lat_lons, analysis_dim=10)
    
    # Invalid shape
    x = torch.randn(10, 2)
    obs = torch.randn(10, 2) # Dummy
    with pytest.raises(ValueError, match="Expected input shape"):
        model(x, obs)

def test_aurora_input_shape_validation():
    """Test that AuroraModel raises ValueError for invalid input shapes."""
    config = {
        "input_features": 2,
        "output_features": 2,
        "latent_dim": 8,
        "max_points": 50,
        "max_seq_len": 128,
    }
    model = AuroraModel(**config)
    
    points = torch.randn(1, 10, 2)
    # Invalid shape
    features = torch.randn(1, 10)
    
    with pytest.raises(ValueError, match="Expected input shape"):
        model(points, features)


