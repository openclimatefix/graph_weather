
import numpy as np
import torch
from graph_weather.models.gencast import Denoiser

def test_gencast_denoiser():
    grid_lat = np.arange(-90, 90, 1)
    grid_lon = np.arange(0, 360, 1)
    input_features_dim = 10
    output_features_dim = 5
    batch_size = 3
    device = "cpu"
    denoiser = Denoiser(
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        input_features_dim=input_features_dim,
        output_features_dim=output_features_dim,
        hidden_dims=[32, 32],
        num_blocks=8,
        num_heads=4,
        splits=2,
        num_hops=1,
        device=device,
    ).to(device)

    corrupted_targets = torch.randn((batch_size, len(grid_lon), len(grid_lat), output_features_dim))
    prev_inputs = torch.randn((batch_size, len(grid_lon), len(grid_lat), 2*input_features_dim))
    noise_levels = torch.rand((batch_size, 1))


    
    preds = denoiser(corrupted_targets=corrupted_targets, 
                         prev_inputs=prev_inputs, 
                         noise_levels=noise_levels)
    loss = torch.mean(preds)
    loss.backward()
 
if __name__ == "__main__":
    test_gencast_denoiser()