"""
WeatherMesh-4 implementation.

Based on the WeatherMesh-4 technical blog: https://windbornesystems.com/blog/weathermesh-4-technical-blog

WeatherMesh-4 succeeds WeatherMesh-3 and introduces a HEALPix mesh to represent the Earth's
spherical surface more faithfully than a latlon grid. Key properties:

- Hierarchical: mesh resolution controlled by a "depth" parameter; each pixel divides into four
  at the next depth.
- Equal area: every pixel represents the same area (unlike latlon).
- IsoLatitude: nodes organized along constant-latitude rings with well-defined neighbors in
  8 directions, so the mesh can be "flattened" and computed on efficiently.

In WM-4, the weather state lives on a HEALPix mesh of depth 8 (~0.25-degree resolution) while the
latent space lives on a depth 5 mesh (~1.8-degree resolution). The encoder down-samples from the
weather resolution to the latent resolution, the processor runs in latent space, and the decoder
up-samples back to the weather resolution.

This implementation reuses the Neighborhood Attention (NATTEN) encoder/processor/decoder layers
from WeatherMesh-3 and adds the same latent rollout, ensemble, and matepoint features as WM-6.
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
class WeatherMesh4Config:
    """Configuration for WeatherMesh-4."""

    # HEALPix mesh resolutions. Weather state at depth 8 (~0.25-deg), latent at depth 5 (~1.8-deg).
    weather_resolution: float = 0.25
    latent_resolution: float = 1.8
    mesh_type: str = "healpix"

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
    def from_json(json: dict) -> "WeatherMesh4Config":
        """Build a config from a JSON-compatible dict."""
        return dacite.from_dict(
            data_class=WeatherMesh4Config, data=json, config=dacite.Config(cast=[tuple])
        )

    def to_json(self) -> dict:
        """Serialize the config to a JSON-compatible dict."""
        return asdict(self)


@dataclass
class WeatherMesh4Output:
    """Output of a WeatherMesh-4 forward pass."""

    surface: torch.Tensor
    pressure: torch.Tensor


def _checkpoint(module: nn.Module, x: torch.Tensor, use_matepoint: bool) -> torch.Tensor:
    """Run a module forward, optionally offloading activations to CPU via matepoint."""
    if use_matepoint:
        from matepoint import checkpoint

        return checkpoint(module, x, use_reentrant=False)
    return module(x)


class WeatherMesh4(nn.Module):
    """WeatherMesh-4 global weather model on a HEALPix mesh."""

    def __init__(
        self,
        weather_resolution: float = 0.25,
        latent_resolution: float = 1.8,
        mesh_type: str = "healpix",
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
        """Initialize the WeatherMesh-4 encoder, processor, and decoder."""
        super().__init__()
        self.weather_resolution = weather_resolution
        self.latent_resolution = latent_resolution
        self.mesh_type = mesh_type
        self.timesteps = timesteps if timesteps is not None else [1, 6]
        self.ensemble_size = ensemble_size if ensemble_size is not None else 1
        self.use_matepoint = use_matepoint if use_matepoint is not None else False

        # Encoder down-samples from the weather resolution to the latent resolution
        # (HEALPix depth 8 -> depth 5) via its strided ConvDownBlocks.
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

        # Decoder up-samples from the latent resolution back to the weather resolution.
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
    ) -> WeatherMesh4Output:
        """
        Latent rollout forecast.

        Encodes the weather-state input down to latent resolution once, steps the processor(s)
        in latent space for ``forecast_steps``, then decodes back up to weather resolution once.
        When ``ensemble_size > 1``, the latent state is replicated with small perturbations and
        each member is rolled out independently.

        Args:
            surface: Surface (2D) input at weather resolution [B, C_s, H, W].
            pressure: Pressure-level (3D) input at weather resolution [B, C_p, D, H, W].
            forecast_steps: Number of autoregressive steps to roll out in latent space.
            ensemble_size: Number of ensemble members. Defaults to config value.

        Returns:
            WeatherMesh4Output with surface and pressure predictions at weather resolution.
            When ensembling, the leading dimension is [B, E, ...].
        """
        if ensemble_size is None:
            ensemble_size = self.ensemble_size

        # Encode to latent space [B, D, H, W, C] (latent resolution)
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

        # Decode each ensemble member back to weather resolution
        if ensemble_size > 1:
            b, e, d, h, w, c = latent.shape
            latent_flat = latent.reshape(b * e, d, h, w, c)
            surface_out, pressure_out = self.decoder(latent_flat)
            surface_out = surface_out.reshape(b, e, *surface_out.shape[1:])
            pressure_out = pressure_out.reshape(b, e, *pressure_out.shape[1:])
        else:
            surface_out, pressure_out = self.decoder(latent)

        return WeatherMesh4Output(surface=surface_out, pressure=pressure_out)
