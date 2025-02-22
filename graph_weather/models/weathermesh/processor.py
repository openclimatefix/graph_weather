"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""
import torch.nn as nn
from natten import NeighborhoodAttention3D


class WeatherMeshProcessor(nn.Module):
    def __init__(self, latent_dim, n_layers=10, kernel=(5,7,7), num_heads=8):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                NeighborhoodAttention3D(
                    dim=latent_dim,
                    num_heads=num_heads,
                    kernel_size=kernel,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
