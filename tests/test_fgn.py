import numpy as np
import pytest
import torch
from packaging.version import Version
from torch_geometric.transforms import TwoHop

from graph_weather.models.fgn import FunctionalGenerativeNetwork


def test_fgn_forward():
    grid_lat = np.arange(-90, 90, 1)
    grid_lon = np.arange(0, 360, 1)
    input_features_dim = 10
    output_features_dim = 5
    batch_size = 3

    model = FunctionalGenerativeNetwork(
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        input_features_dim=input_features_dim,
        output_features_dim=output_features_dim,
        noise_dimension=32,
        hidden_dims=[16, 32],
        num_blocks=3,
        num_heads=4,
        splits=0,
        num_hops=1,
        device=torch.device("cpu"),
    ).eval()

    prev_inputs = torch.randn((batch_size, len(grid_lon), len(grid_lat), 2 * input_features_dim))

    with torch.no_grad():
        preds = model(previous_weather_state=prev_inputs)

    assert not torch.isnan(preds).any()
    assert preds.shape == (1, 2, len(grid_lon), len(grid_lat), output_features_dim)
