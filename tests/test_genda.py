import numpy as np
import torch

from graph_weather.models.genda.model import GenDA


def test_genda_forward():

    model = GenDA(
        grid_lon=np.linspace(0, 1, 2),
        grid_lat=np.linspace(0, 1, 2),
        input_features_dim=2,
        output_features_dim=2,
        hidden_dims=[16, 16],
        num_blocks=1,
        num_heads=1,
        splits=1,
        num_hops=1,
    )

    corrupted_targets = torch.randn(1, 2, 2, 2)
    prev_inputs = torch.randn(1, 2, 2, 4)
    sensor_mask = torch.ones(1, 2, 2, 1)
    sensor_values = torch.randn(1, 2, 2, 1)

    noise = torch.ones(1, 1)

    out = model(
        corrupted_targets=corrupted_targets,
        prev_inputs=prev_inputs,
        noise_levels=noise,
        sensor_mask=sensor_mask,
        sensor_values=sensor_values,
    )

    assert out.shape == corrupted_targets.shape
