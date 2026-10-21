"""
WeatherMesh-6 implementation.

Based on the WeatherMesh-6 announcement: https://windbornesystems.com/blog/introducing-wm-6

WeatherMesh-6 is a global weather model operating at 0.25-degree resolution that models a full
ensemble in latent space. This implementation reuses the Neighborhood Attention (NATTEN)
encoder/processor/decoder layers from WeatherMesh-3 and adds:

- Latent rollout: encode once, step the processor in latent space for N forecast steps, decode once.
- Ensemble members: replicate the latent state with perturbations and roll out each member.
- Matepoint activation offloading: offload processor activations to CPU RAM to enable longer
  rollouts on limited GPU VRAM (https://github.com/windborne/matepoint).
"""

from dataclasses import asdict, dataclass
from typing import List, Optional

import dacite
import torch
import torch.nn as nn

from graph_weather.models.weathermesh.decoder import WeatherMeshDecoder
from graph_weather.models.weathermesh.encoder import WeatherMeshEncoder
from graph_weather.models.weathermesh.processor import WeatherMeshProcessor


@dataclass
class WeatherMesh6Config:
    """Configuration for WeatherMesh-6."""

    resolution: float = 0.25
    surface_channels: int = 0
    pressure_channels: int = 0
    pressure_levels: int = 25
    latent_dim: int = 256
    hidden_dim: int = 256
    kernel: tuple = (5, 7, 7)
    num_heads: int = 8
    encoder_num_conv_blocks: int = 3
    encoder_num_transformer_layers: int = 3
    encoder_hidden_dim: int = 256
    decoder_num_conv_blocks: int = 3
    decoder_num_transformer_layers: int = 3
    decoder_hidden_dim: int = 256
    processor_num_layers: int = 10
    timesteps: List[int] = None
    ensemble_size: int = 1
    use_matepoint: bool = False

    def __post_init__(self):
        if self.timesteps is None:
            self.timesteps = [1, 6]

    @staticmethod
    def from_json(json: dict) -> "WeatherMesh6Config":
        """Build a config from a JSON-compatible dict."""
        return dacite.from_dict(
            data_class=WeatherMesh6Config, data=json, config=dacite.Config(cast=[tuple])
        )

    def to_json(self) -> dict:
        """Serialize the config to a JSON-compatible dict."""
        return asdict(self)


@dataclass
class WeatherMesh6Output:
    """Output of a WeatherMesh-6 forward pass."""

    surface: torch.Tensor
    pressure: torch.Tensor


def _checkpoint(module: nn.Module, x: torch.Tensor, use_matepoint: bool) -> torch.Tensor:
    """Run a module forward, optionally offloading activations to CPU via matepoint."""
    if use_matepoint:
        from matepoint import checkpoint

        return checkpoint(module, x, use_reentrant=False)
    return module(x)


class WeatherMesh6(nn.Module):
    """WeatherMesh-6 global weather model with latent-space ensembles and latent rollout."""

    def __init__(
        self,
        resolution: float = 0.25,
        surface_channels: int | None = None,
        pressure_channels: int | None = None,
        pressure_levels: int | None = 25,
        latent_dim: int | None = 256,
        hidden_dim: int | None = 256,
        kernel: tuple | None = (5, 7, 7),
        num_heads: int | None = 8,
        encoder_num_conv_blocks: int | None = 3,
        encoder_num_transformer_layers: int | None = 3,
        encoder_hidden_dim: int | None = 256,
        decoder_num_conv_blocks: int | None = 3,
        decoder_num_transformer_layers: int | None = 3,
        decoder_hidden_dim: int | None = 256,
        processor_num_layers: int | None = 10,
        timesteps: List[int] | None = None,
        ensemble_size: int | None = 1,
        use_matepoint: bool | None = False,
    ):
        """Initialize the WeatherMesh-6 encoder, processor, and decoder."""
        super().__init__()
        self.resolution = resolution
        self.timesteps = timesteps if timesteps is not None else [1, 6]
        self.ensemble_size = ensemble_size if ensemble_size is not None else 1
        self.use_matepoint = use_matepoint if use_matepoint is not None else False

        self.encoder = WeatherMeshEncoder(
            input_channels_2d=surface_channels,
            input_channels_3d=pressure_channels,
            latent_dim=latent_dim,
            n_pressure_levels=pressure_levels,
            num_conv_blocks=encoder_num_conv_blocks,
            hidden_dim=encoder_hidden_dim,
            kernel_size=kernel,
            num_heads=num_heads,
            num_transformer_layers=encoder_num_transformer_layers,
        )

        self.processors = nn.ModuleList(
            [
                WeatherMeshProcessor(
                    latent_dim=latent_dim,
                    n_layers=processor_num_layers,
                    kernel=kernel,
                    num_heads=num_heads,
                )
                for _ in range(len(self.timesteps))
            ]
        )

        self.decoder = WeatherMeshDecoder(
            latent_dim=latent_dim,
            output_channels_2d=surface_channels,
            output_channels_3d=pressure_channels,
            n_conv_blocks=decoder_num_conv_blocks,
            hidden_dim=decoder_hidden_dim,
            kernel_size=kernel,
            num_heads=num_heads,
            num_transformer_layers=decoder_num_transformer_layers,
        )

    def forward(
        self,
        surface: torch.Tensor,
        pressure: torch.Tensor,
        forecast_steps: int,
        ensemble_size: Optional[int] = None,
    ) -> WeatherMesh6Output:
        """
        Latent rollout forecast.

        Encodes the input once, steps the processor(s) in latent space for ``forecast_steps``,
        then decodes once. When ``ensemble_size > 1``, the latent state is replicated with small
        perturbations and each member is rolled out independently.

        Args:
            surface: Surface (2D) input [B, C_s, H, W].
            pressure: Pressure-level (3D) input [B, C_p, D, H, W].
            forecast_steps: Number of autoregressive steps to roll out in latent space.
            ensemble_size: Number of ensemble members. Defaults to config value.

        Returns:
            WeatherMesh6Output with surface and pressure predictions. When ensembling, the
            leading dimension is [B, E, ...].
        """
        if ensemble_size is None:
            ensemble_size = self.ensemble_size

        # Encode to latent space [B, D, H, W, C]
        latent = self.encoder(surface, pressure)

        # Replicate into ensemble members with small perturbations
        if ensemble_size > 1:
            latent = latent.unsqueeze(1).repeat(1, ensemble_size, 1, 1, 1, 1)
            latent = latent + torch.randn_like(latent) * 0.01

        # Latent rollout: step processors in latent space without decoding each step
        for _ in range(forecast_steps):
            for processor in self.processors:
                if ensemble_size > 1:
                    b, e, d, h, w, c = latent.shape
                    latent = latent.reshape(b * e, d, h, w, c)
                    latent = _checkpoint(processor, latent, self.use_matepoint)
                    latent = latent.reshape(b, e, d, h, w, c)
                else:
                    latent = _checkpoint(processor, latent, self.use_matepoint)

        # Decode each ensemble member
        if ensemble_size > 1:
            b, e, d, h, w, c = latent.shape
            latent_flat = latent.reshape(b * e, d, h, w, c)
            surface_out, pressure_out = self.decoder(latent_flat)
            surface_out = surface_out.reshape(b, e, *surface_out.shape[1:])
            pressure_out = pressure_out.reshape(b, e, *pressure_out.shape[1:])
        else:
            surface_out, pressure_out = self.decoder(latent)

        return WeatherMesh6Output(surface=surface_out, pressure=pressure_out)
