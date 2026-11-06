"""Wrapper adapters that make existing tensor-based components work with PyG Data protocols.

These wrappers allow the existing Encoder, Processor, and Decoder classes
(which use raw tensors) to be composed in a WeatherModel via the protocol
interfaces, without modifying the original component code.

The PyG Data objects use the `batch_size` attribute to track the number of
samples in a batch, as the flat [B*N, F] format cannot infer B from N alone.
"""

from typing import Tuple

import einops
import torch
from torch_geometric.data import Data

from graph_weather.models.aurora.model import (
    PointCloudProcessor as AuroraProcessor,
)
from graph_weather.models.aurora.model import (
    PointDecoder as AuroraDecoder,
)
from graph_weather.models.aurora.model import (
    PointEncoder as AuroraEncoder,
)
from graph_weather.models.cafa.decoder import CaFADecoder as CaFADecoderTensor
from graph_weather.models.cafa.encoder import CaFAEncoder as CaFAEncoderTensor
from graph_weather.models.cafa.processor import CaFAProcessor as CaFAProcessorTensor
from graph_weather.models.layers.decoder import Decoder as TensorDecoder
from graph_weather.models.layers.encoder import Encoder as TensorEncoder
from graph_weather.models.layers.processor import Processor as TensorProcessor
from graph_weather.models.protocols import DecoderProtocol, EncoderProtocol, ProcessorProtocol
from graph_weather.models.weathermesh.decoder import WeatherMeshDecoder as WMDecoderTensor
from graph_weather.models.weathermesh.encoder import WeatherMeshEncoder as WMEncoderTensor
from graph_weather.models.weathermesh.processor import WeatherMeshProcessor as WMProcessorTensor


def _get_batch_size(data: Data) -> int:
    """Extract batch size from a PyG Data object.

    Checks the `batch_size` attribute first, then falls back to the PyG
    `batch` attribute, and finally defaults to 1.
    """
    if hasattr(data, "batch_size") and data.batch_size is not None:
        return int(data.batch_size)
    if hasattr(data, "batch") and data.batch is not None:
        return int(data.batch.max()) + 1
    return 1


class EncoderWrapper(torch.nn.Module, EncoderProtocol):
    """Wraps the tensor-based Encoder to work with PyG Data objects.

    The existing Encoder accepts a feature tensor [B, N, F] and returns
    a tuple of (latent_features, edge_index, edge_attr). This wrapper
    converts between PyG Data and raw tensors.
    """

    def __init__(self, encoder: TensorEncoder):
        """
        Initialize the wrapper.

        Args:
            encoder: An existing Encoder instance.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, data: Data) -> Data:
        """Encode input graph data to latent graph data.

        Args:
            data: PyG Data with `x` of shape [B*N, F] and `batch_size` attribute.

        Returns:
            PyG Data with encoded latent node features, edge_index, and edge_attr.
        """
        x = data.x
        batch_size = _get_batch_size(data)
        features = einops.rearrange(x, "(b n) f -> b n f", b=batch_size)

        # Call the existing encoder
        latent_features, edge_index, edge_attr = self.encoder(features)

        # Return as PyG Data, preserving batch_size
        result = Data(x=latent_features, edge_index=edge_index, edge_attr=edge_attr)
        result.batch_size = batch_size
        return result


class ProcessorWrapper(torch.nn.Module, ProcessorProtocol):
    """Wraps the tensor-based Processor to work with PyG Data objects.

    The existing Processor accepts (x, edge_index, edge_attr) tensors and
    returns updated node features. This wrapper converts between PyG Data
    and raw tensors.
    """

    def __init__(self, processor: TensorProcessor):
        """
        Initialize the wrapper.

        Args:
            processor: An existing Processor instance.
        """
        super().__init__()
        self.processor = processor

    def forward(self, data: Data) -> Data:
        """Process latent graph data.

        Args:
            data: PyG Data with node features `x`, `edge_index`, and `edge_attr`.

        Returns:
            PyG Data with processed node features.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch_size = _get_batch_size(data)

        # Call the existing processor
        out = self.processor(x, edge_index, edge_attr, batch_size=batch_size)

        result = Data(x=out, edge_index=edge_index, edge_attr=edge_attr)
        result.batch_size = batch_size
        return result


class DecoderWrapper(torch.nn.Module, DecoderProtocol):
    """Wraps the tensor-based Decoder to work with PyG Data objects.

    The existing Decoder accepts (processed_features, original_features) tensors
    and returns output features. This wrapper converts between PyG Data and
    raw tensors, preserving the residual (skip) connection.
    """

    def __init__(self, decoder: TensorDecoder):
        """
        Initialize the wrapper.

        Args:
            decoder: An existing Decoder instance.
        """
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: Data, original: Data) -> Data:
        """Decode latent graph data back to output graph data.

        Args:
            latent: Processed latent graph from the processor.
            original: Original input graph (for skip connection).

        Returns:
            PyG Data with decoded node features matching the original graph structure.
        """
        latent_x = latent.x
        original_x = original.x
        batch_size = _get_batch_size(original)

        # Reshape original features to [B, N, F] for the decoder
        original_features = einops.rearrange(original_x, "(b n) f -> b n f", b=batch_size)

        # Call the existing decoder
        out = self.decoder(latent_x, original_features)

        # Reshape back to [B*N, F]
        out = einops.rearrange(out, "b n f -> (b n) f")

        result = Data(x=out)
        result.batch_size = batch_size
        return result


def build_graphcast_stage(
    lat_lons: list,
    resolution: int = 2,
    input_dim: int = 78,
    output_dim: int = 78,
    hidden_dim: int = 256,
    num_processor_blocks: int = 9,
    hidden_layers: int = 2,
    mlp_norm_type: str = "LayerNorm",
    use_checkpointing: bool = False,
    efficient_batching: bool = False,
) -> Tuple[EncoderWrapper, ProcessorWrapper, DecoderWrapper]:
    """Convenience function to build a GraphCast-style stage with wrapped components.

    Args:
        lat_lons: List of (lat, lon) tuples.
        resolution: H3 resolution level.
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        hidden_dim: Hidden dimension for all layers.
        num_processor_blocks: Number of message passing blocks.
        hidden_layers: Number of hidden layers in MLPs.
        mlp_norm_type: Normalization type for MLPs.
        use_checkpointing: Enable gradient checkpointing.
        efficient_batching: Use efficient batching.

    Returns:
        Tuple of (encoder, processor, decoder) wrapped components.
    """
    encoder = TensorEncoder(
        lat_lons=lat_lons,
        resolution=resolution,
        input_dim=input_dim,
        output_dim=hidden_dim,
        output_edge_dim=hidden_dim,
        hidden_dim_processor_node=hidden_dim,
        hidden_dim_processor_edge=hidden_dim,
        hidden_layers_processor_node=hidden_layers,
        hidden_layers_processor_edge=hidden_layers,
        mlp_norm_type=mlp_norm_type,
        use_checkpointing=use_checkpointing,
        efficient_batching=efficient_batching,
    )

    processor = TensorProcessor(
        input_dim=hidden_dim,
        edge_dim=hidden_dim,
        num_blocks=num_processor_blocks,
        hidden_dim_processor_node=hidden_dim,
        hidden_dim_processor_edge=hidden_dim,
        hidden_layers_processor_node=hidden_layers,
        hidden_layers_processor_edge=hidden_layers,
        mlp_norm_type=mlp_norm_type,
        use_checkpointing=use_checkpointing,
    )

    decoder = TensorDecoder(
        lat_lons=lat_lons,
        resolution=resolution,
        input_dim=hidden_dim,
        output_dim=output_dim,
        hidden_dim_processor_node=hidden_dim,
        hidden_dim_processor_edge=hidden_dim,
        hidden_layers_processor_node=hidden_layers,
        hidden_layers_processor_edge=hidden_layers,
        mlp_norm_type=mlp_norm_type,
        hidden_dim_decoder=hidden_dim,
        hidden_layers_decoder=hidden_layers,
        use_checkpointing=use_checkpointing,
        efficient_batching=efficient_batching,
    )

    return EncoderWrapper(encoder), ProcessorWrapper(processor), DecoderWrapper(decoder)


# ---------------------------------------------------------------------------
# Aurora adapters
# ---------------------------------------------------------------------------


class AuroraEncoderWrapper(torch.nn.Module, EncoderProtocol):
    """Wraps Aurora PointEncoder to work with PyG Data objects.

    Aurora's encoder takes (points, features) of shape [B, N, 2] and [B, N, F]
    and returns embeddings [B, N, embed_dim]. The PyG Data object stores
    coordinates in `pos` and features in `x`.
    """

    def __init__(self, encoder: AuroraEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, data: Data) -> Data:
        x = data.x  # [B*N, F]
        pos = data.pos  # [B*N, 2] — lat/lon coordinates
        batch_size = _get_batch_size(data)

        features = einops.rearrange(x, "(b n) f -> b n f", b=batch_size)
        points = einops.rearrange(pos, "(b n) d -> b n d", b=batch_size)

        out = self.encoder(points, features)  # [B, N, embed_dim]
        out = einops.rearrange(out, "b n f -> (b n) f")

        result = Data(x=out, pos=pos)
        result.batch_size = batch_size
        return result


class AuroraProcessorWrapper(torch.nn.Module, ProcessorProtocol):
    """Wraps Aurora PointCloudProcessor to work with PyG Data objects."""

    def __init__(self, processor: AuroraProcessor):
        super().__init__()
        self.processor = processor

    def forward(self, data: Data) -> Data:
        x = data.x
        batch_size = _get_batch_size(data)
        features = einops.rearrange(x, "(b n) f -> b n f", b=batch_size)
        out = self.processor(features)
        out = einops.rearrange(out, "b n f -> (b n) f")

        result = Data(x=out, pos=data.pos)
        result.batch_size = batch_size
        return result


class AuroraDecoderWrapper(torch.nn.Module, DecoderProtocol):
    """Wraps Aurora PointDecoder to work with PyG Data objects."""

    def __init__(self, decoder: AuroraDecoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: Data, original: Data) -> Data:
        x = latent.x
        batch_size = _get_batch_size(latent)
        features = einops.rearrange(x, "(b n) f -> b n f", b=batch_size)
        out = self.decoder(features)
        out = einops.rearrange(out, "b n f -> (b n) f")

        result = Data(x=out, pos=original.pos if hasattr(original, "pos") else None)
        result.batch_size = batch_size
        return result


def build_aurora_stage(
    input_features: int,
    output_features: int,
    latent_dim: int = 256,
    num_layers: int = 4,
    max_seq_len: int = 1024,
) -> Tuple[AuroraEncoderWrapper, AuroraProcessorWrapper, AuroraDecoderWrapper]:
    """Build an Aurora-style stage with wrapped components."""
    encoder = AuroraEncoder(input_features, latent_dim, max_seq_len)
    processor = AuroraProcessor(latent_dim, num_layers)
    decoder = AuroraDecoder(latent_dim, output_features)
    return (
        AuroraEncoderWrapper(encoder),
        AuroraProcessorWrapper(processor),
        AuroraDecoderWrapper(decoder),
    )


# ---------------------------------------------------------------------------
# CAFA adapters
# ---------------------------------------------------------------------------


class CaFAEncoderWrapper(torch.nn.Module, EncoderProtocol):
    """Wraps CAFA encoder (image-based) to work with PyG Data.

    CAFA operates on 2D image tensors [B, C, H, W]. The PyG Data object
    stores the image in the `img` attribute.
    """

    def __init__(self, encoder: CaFAEncoderTensor):
        super().__init__()
        self.encoder = encoder

    def forward(self, data: Data) -> Data:
        img = data.img  # [B, C, H, W]
        out = self.encoder(img)  # [B, model_dim, h, w] (downsampled)
        result = Data(img=out)
        result.batch_size = img.shape[0]
        return result


class CaFAProcessorWrapper(torch.nn.Module, ProcessorProtocol):
    """Wraps CAFA processor to work with PyG Data."""

    def __init__(self, processor: CaFAProcessorTensor):
        super().__init__()
        self.processor = processor

    def forward(self, data: Data) -> Data:
        img = data.img
        out = self.processor(img)
        result = Data(img=out)
        result.batch_size = img.shape[0]
        return result


class CaFADecoderWrapper(torch.nn.Module, DecoderProtocol):
    """Wraps CAFA decoder to work with PyG Data."""

    def __init__(self, decoder: CaFADecoderTensor):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: Data, original: Data) -> Data:
        img = latent.img
        out = self.decoder(img)
        result = Data(img=out)
        result.batch_size = img.shape[0]
        return result


def build_cafa_stage(
    input_channels: int,
    output_channels: int,
    model_dim: int = 256,
    downsampling_factor: int = 2,
    processor_depth: int = 6,
    num_heads: int = 8,
    dim_head: int = 64,
    feedforward_multiplier: int = 4,
    dropout: float = 0.0,
) -> Tuple[CaFAEncoderWrapper, CaFAProcessorWrapper, CaFADecoderWrapper]:
    """Build a CAFA-style stage with wrapped components."""
    encoder = CaFAEncoderTensor(input_channels, model_dim, downsampling_factor)
    processor = CaFAProcessorTensor(
        model_dim, processor_depth, num_heads, dim_head, feedforward_multiplier, dropout
    )
    decoder = CaFADecoderTensor(model_dim, output_channels, downsampling_factor)
    return (
        CaFAEncoderWrapper(encoder),
        CaFAProcessorWrapper(processor),
        CaFADecoderWrapper(decoder),
    )


# ---------------------------------------------------------------------------
# WeatherMesh adapters
# ---------------------------------------------------------------------------


class WeatherMeshEncoderWrapper(torch.nn.Module, EncoderProtocol):
    """Wraps WeatherMesh encoder to work with PyG Data.

    WeatherMesh operates on 2D/3D field tensors. The PyG Data object stores
    the fields in the `field_2d` and `field_3d` attributes.
    """

    def __init__(self, encoder: WMEncoderTensor):
        super().__init__()
        self.encoder = encoder

    def forward(self, data: Data) -> Data:
        field_2d = data.field_2d  # [B, C, H, W]
        field_3d = data.field_3d  # [B, C, D, H, W]
        out = self.encoder(field_2d, field_3d)  # latent tensor
        result = Data(latent=out)
        result.batch_size = field_2d.shape[0]
        return result


class WeatherMeshProcessorWrapper(torch.nn.Module, ProcessorProtocol):
    """Wraps WeatherMesh processor to work with PyG Data."""

    def __init__(self, processor: WMProcessorTensor):
        super().__init__()
        self.processor = processor

    def forward(self, data: Data) -> Data:
        latent = data.latent
        out = self.processor(latent)
        result = Data(latent=out)
        result.batch_size = latent.shape[0]
        return result


class WeatherMeshDecoderWrapper(torch.nn.Module, DecoderProtocol):
    """Wraps WeatherMesh decoder to work with PyG Data."""

    def __init__(self, decoder: WMDecoderTensor):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: Data, original: Data) -> Data:
        out = self.decoder(latent.latent)
        result = Data(field_2d=out.get("field_2d", None), field_3d=out.get("field_3d", None))
        result.batch_size = latent.latent.shape[0]
        return result


def build_weathermesh_stage(
    input_channels_2d: int,
    input_channels_3d: int,
    output_channels_2d: int,
    output_channels_3d: int,
    latent_dim: int = 256,
    n_layers: int = 10,
) -> Tuple[WeatherMeshEncoderWrapper, WeatherMeshProcessorWrapper, WeatherMeshDecoderWrapper]:
    """Build a WeatherMesh-style stage with wrapped components."""
    encoder = WMEncoderTensor(input_channels_2d, input_channels_3d, latent_dim)
    processor = WMProcessorTensor(latent_dim, n_layers)
    decoder = WMDecoderTensor(latent_dim, output_channels_2d, output_channels_3d)
    return (
        WeatherMeshEncoderWrapper(encoder),
        WeatherMeshProcessorWrapper(processor),
        WeatherMeshDecoderWrapper(decoder),
    )
