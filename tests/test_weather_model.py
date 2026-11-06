"""Tests for the WeatherModel orchestrator and protocol-based components."""

import h3
import torch
from torch_geometric.data import Data

from graph_weather.models import (
    DecoderWrapper,
    EncoderWrapper,
    ProcessorWrapper,
    StageConfig,
    WeatherModel,
    WeatherModelConfig,
    build_graphcast_stage,
)
from graph_weather.models.layers.decoder import Decoder
from graph_weather.models.layers.encoder import Encoder
from graph_weather.models.layers.processor import Processor


def _make_lat_lons():
    """Create a small grid of lat/lon points for testing."""
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))
    return lat_lons


def test_single_stage_weather_model():
    """Test WeatherModel with a single GraphCast-style stage."""
    lat_lons = _make_lat_lons()
    encoder, processor, decoder = build_graphcast_stage(
        lat_lons, hidden_dim=64, num_processor_blocks=2, hidden_layers=1
    )
    stage = StageConfig(encoder, processor, decoder)
    model = WeatherModel(stage).eval()

    batch_size = 2
    features = torch.randn((batch_size, len(lat_lons), 78))
    # Flatten to [B*N, F] for PyG Data
    x = features.reshape(-1, 78)
    data = Data(x=x, batch_size=batch_size)

    with torch.no_grad():
        out = model(data)
    assert out.x.size() == (batch_size * len(lat_lons), 78)


def test_multi_stage_weather_model():
    """Test WeatherModel with two stages."""
    lat_lons = _make_lat_lons()
    enc1, proc1, dec1 = build_graphcast_stage(
        lat_lons, hidden_dim=64, num_processor_blocks=2, hidden_layers=1
    )
    enc2, proc2, dec2 = build_graphcast_stage(
        lat_lons, hidden_dim=64, num_processor_blocks=2, hidden_layers=1
    )
    stage1 = StageConfig(enc1, proc1, dec1)
    stage2 = StageConfig(enc2, proc2, dec2)
    model = WeatherModel([stage1, stage2]).eval()

    batch_size = 2
    features = torch.randn((batch_size, len(lat_lons), 78))
    x = features.reshape(-1, 78)
    data = Data(x=x, batch_size=batch_size)

    with torch.no_grad():
        out = model(data)
    assert out.x.size() == (batch_size * len(lat_lons), 78)


def test_weather_model_checkpointing():
    """Test WeatherModel with checkpointing enabled."""
    lat_lons = _make_lat_lons()
    encoder, processor, decoder = build_graphcast_stage(
        lat_lons, hidden_dim=64, num_processor_blocks=2, hidden_layers=1
    )
    stage = StageConfig(
        encoder, processor, decoder,
        checkpoint_encoder=True,
        checkpoint_processor=True,
        checkpoint_decoder=True,
    )
    model = WeatherModel(stage)

    batch_size = 1
    features = torch.randn((batch_size, len(lat_lons), 78))
    x = features.reshape(-1, 78)
    data = Data(x=x, batch_size=batch_size)

    out = model(data)
    loss = out.x.sum()
    loss.backward()
    # Check that gradients exist
    assert encoder.encoder.node_encoder.model[0].weight.grad is not None


def test_weather_model_full_checkpointing():
    """Test WeatherModel with full model-level checkpointing."""
    lat_lons = _make_lat_lons()
    encoder, processor, decoder = build_graphcast_stage(
        lat_lons, hidden_dim=64, num_processor_blocks=2, hidden_layers=1
    )
    stage = StageConfig(encoder, processor, decoder)
    model = WeatherModel(stage, checkpoint_model=True)

    batch_size = 1
    features = torch.randn((batch_size, len(lat_lons), 78))
    x = features.reshape(-1, 78)
    data = Data(x=x, batch_size=batch_size)

    out = model(data)
    loss = out.x.sum()
    loss.backward()
    assert encoder.encoder.node_encoder.model[0].weight.grad is not None


def test_weather_model_config_strategies():
    """Test WeatherModelConfig helper methods."""
    lat_lons = _make_lat_lons()
    encoder, processor, decoder = build_graphcast_stage(
        lat_lons, hidden_dim=64, num_processor_blocks=2, hidden_layers=1
    )
    stage = StageConfig(encoder, processor, decoder)
    model = WeatherModel(stage)

    # Test no checkpointing
    WeatherModelConfig.no_checkpointing(model)
    assert not model.checkpoint_model

    # Test full checkpointing
    WeatherModelConfig.full_checkpointing(model)
    assert model.checkpoint_model

    # Test balanced checkpointing
    WeatherModelConfig.balanced_checkpointing(model)
    assert not model.checkpoint_model
    assert model._checkpoint_encoder[0] is True
    assert model._checkpoint_processor[0] is True
    assert model._checkpoint_decoder[0] is True


def test_encoder_wrapper():
    """Test EncoderWrapper converts PyG Data to/from tensors correctly."""
    lat_lons = _make_lat_lons()
    encoder = Encoder(lat_lons, output_dim=64).eval()
    wrapper = EncoderWrapper(encoder)

    batch_size = 2
    features = torch.randn((batch_size, len(lat_lons), 78))
    x = features.reshape(-1, 78)
    data = Data(x=x, batch_size=batch_size)

    with torch.no_grad():
        out = wrapper(data)
    num_h3 = h3.get_num_cells(2)
    assert out.x.size() == (batch_size * num_h3, 64)
    assert out.edge_index is not None
    assert out.edge_attr is not None


def test_processor_wrapper():
    """Test ProcessorWrapper converts PyG Data to/from tensors correctly."""
    lat_lons = _make_lat_lons()
    encoder = Encoder(lat_lons, output_dim=64, output_edge_dim=64).eval()
    processor = Processor(input_dim=64, edge_dim=64, num_blocks=2).eval()
    wrapper = ProcessorWrapper(processor)

    batch_size = 2
    features = torch.randn((batch_size, len(lat_lons), 78))
    with torch.no_grad():
        latent_features, edge_index, edge_attr = encoder(features)
    data = Data(x=latent_features, edge_index=edge_index, edge_attr=edge_attr, batch_size=batch_size)

    with torch.no_grad():
        out = wrapper(data)
    assert out.x.size() == latent_features.size()


def test_decoder_wrapper():
    """Test DecoderWrapper converts PyG Data to/from tensors correctly."""
    lat_lons = _make_lat_lons()
    decoder = Decoder(lat_lons, input_dim=64, output_dim=78).eval()
    wrapper = DecoderWrapper(decoder)

    batch_size = 2
    num_h3 = h3.get_num_cells(2)
    latent = Data(x=torch.randn((batch_size * num_h3, 64)), batch_size=batch_size)
    original_features = torch.randn((batch_size, len(lat_lons), 78))
    original = Data(x=original_features.reshape(-1, 78), batch_size=batch_size)

    with torch.no_grad():
        out = wrapper(latent, original)
    assert out.x.size() == (batch_size * len(lat_lons), 78)


def test_mixed_components():
    """Test that components from different stages can be mixed."""
    lat_lons = _make_lat_lons()
    # Build two stages with different configs
    enc1, proc1, dec1 = build_graphcast_stage(
        lat_lons, hidden_dim=64, num_processor_blocks=1, hidden_layers=1
    )
    enc2, proc2, dec2 = build_graphcast_stage(
        lat_lons, hidden_dim=64, num_processor_blocks=3, hidden_layers=2
    )

    # Mix: stage1 uses enc1, proc2, dec1; stage2 uses enc2, proc1, dec2
    stage1 = StageConfig(enc1, proc2, dec1)
    stage2 = StageConfig(enc2, proc1, dec2)
    model = WeatherModel([stage1, stage2]).eval()

    batch_size = 1
    features = torch.randn((batch_size, len(lat_lons), 78))
    x = features.reshape(-1, 78)
    data = Data(x=x, batch_size=batch_size)

    with torch.no_grad():
        out = model(data)
    assert out.x.size() == (batch_size * len(lat_lons), 78)
