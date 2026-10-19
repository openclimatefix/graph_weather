"""Training script for WeatherMesh-4.

Trains a WeatherMesh-4 model on surface + pressure-level data with a latent rollout loss.
WM-4 uses a HEALPix mesh: weather state at depth 8 (~0.25-deg), latent at depth 5 (~1.8-deg).
"""

import os
import sys

import numpy as np
import torch
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from graph_weather.models.weathermesh.weathermesh4 import WeatherMesh4, WeatherMesh4Config


class SyntheticWM4Dataset(torch.utils.data.Dataset):
    """Synthetic surface + pressure dataset for smoke-testing WM-4 training."""

    def __init__(self, n_samples=16, surface_channels=4, pressure_channels=2,
                 pressure_levels=5, h=32, w=64):
        self.n_samples = n_samples
        self.surface_channels = surface_channels
        self.pressure_channels = pressure_channels
        self.pressure_levels = pressure_levels
        self.h = h
        self.w = w

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        surface = torch.randn(self.surface_channels, self.h, self.w)
        pressure = torch.randn(self.pressure_channels, self.pressure_levels, self.h, self.w)
        surface_next = torch.randn(self.surface_channels, self.h, self.w)
        pressure_next = torch.randn(self.pressure_channels, self.pressure_levels, self.h, self.w)
        return surface, pressure, surface_next, pressure_next


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = WeatherMesh4Config(
        weather_resolution=0.25,
        latent_resolution=1.8,
        mesh_type="healpix",
        surface_channels=4,
        pressure_channels=2,
        pressure_levels=5,
        latent_dim=32,
        hidden_dim=32,
        kernel=(3, 3, 3),
        num_heads=2,
        encoder_num_conv_blocks=2,
        encoder_num_transformer_layers=2,
        encoder_hidden_dim=32,
        decoder_num_conv_blocks=2,
        decoder_num_transformer_layers=2,
        decoder_hidden_dim=32,
        processor_num_layers=4,
        timesteps=[1, 6],
        ensemble_size=1,
        use_matepoint=False,
    )
    model = WeatherMesh4(**config.to_json()).to(device)

    dataset = SyntheticWM4Dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(3):
        running_loss = 0.0
        for surface, pressure, surface_next, pressure_next in loader:
            surface = surface.to(device)
            pressure = pressure.to(device)
            surface_next = surface_next.to(device)
            pressure_next = pressure_next.to(device)

            optimizer.zero_grad()
            out = model(surface, pressure, forecast_steps=1)
            loss = criterion(out.surface, surface_next) + criterion(out.pressure, pressure_next)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}: loss={running_loss / len(loader):.4f}")

    print("Finished WeatherMesh-4 training")


if __name__ == "__main__":
    main()
