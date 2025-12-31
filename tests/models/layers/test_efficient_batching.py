"""Tests for efficient batching implementation."""

import numpy as np
import pytest
import torch

from graph_weather.models.layers.decoder import Decoder
from graph_weather.models.layers.encoder import Encoder
from graph_weather.models.layers.processor import Processor


def create_lat_lon_grid(resolution_deg: float):
    """Create a lat/lon grid at specified resolution."""
    lat_lons = []
    lats = np.arange(-90, 90, resolution_deg)
    lons = np.arange(0, 360, resolution_deg)
    for lat in lats:
        for lon in lons:
            lat_lons.append((float(lat), float(lon)))
    return lat_lons


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("resolution_deg", [10.0, 5.0])
def test_encoder_efficient_batching(batch_size, resolution_deg):
    """Test that Encoder produces identical outputs with efficient batching."""
    lat_lons = create_lat_lon_grid(resolution_deg)
    num_nodes = len(lat_lons)

    torch.manual_seed(42)
    features = torch.randn((batch_size, num_nodes, 78))

    # Original batching
    encoder_original = Encoder(
        lat_lons, resolution=2, input_dim=78, output_dim=256, efficient_batching=False
    )
    encoder_original.eval()

    with torch.no_grad():
        x_orig, edge_idx_orig, edge_attr_orig = encoder_original(features)

    # Efficient batching
    encoder_efficient = Encoder(
        lat_lons, resolution=2, input_dim=78, output_dim=256, efficient_batching=True
    )
    encoder_efficient.load_state_dict(encoder_original.state_dict())
    encoder_efficient.eval()

    with torch.no_grad():
        x_eff, edge_idx_eff, edge_attr_eff = encoder_efficient(features)

    # Verify outputs match
    assert torch.allclose(x_orig, x_eff, atol=1e-5)

    # Verify graph sharing (edges should not scale with batch size)
    expected_reduction = batch_size
    actual_reduction = edge_idx_orig.shape[1] / edge_idx_eff.shape[1]
    assert abs(actual_reduction - expected_reduction) < 0.1


@pytest.mark.parametrize("batch_size", [1, 2])
def test_decoder_efficient_batching(batch_size):
    """Test that Decoder produces identical outputs with efficient batching."""
    lat_lons = create_lat_lon_grid(resolution_deg=5.0)
    num_nodes = len(lat_lons)

    torch.manual_seed(42)
    num_h3 = 5882
    processor_features = torch.randn((batch_size * num_h3, 256))
    start_features = torch.randn((batch_size, num_nodes, 78))

    # Original batching
    decoder_original = Decoder(
        lat_lons, resolution=2, input_dim=256, output_dim=78, efficient_batching=False
    )
    decoder_original.eval()

    with torch.no_grad():
        out_orig = decoder_original(processor_features, start_features)

    # Efficient batching
    decoder_efficient = Decoder(
        lat_lons, resolution=2, input_dim=256, output_dim=78, efficient_batching=True
    )
    decoder_efficient.load_state_dict(decoder_original.state_dict())
    decoder_efficient.eval()

    with torch.no_grad():
        out_eff = decoder_efficient(processor_features, start_features)

    assert torch.allclose(out_orig, out_eff, atol=1e-5)


def test_full_pipeline():
    """Test the full encoder->processor->decoder pipeline."""
    batch_size = 2
    lat_lons = create_lat_lon_grid(resolution_deg=5.0)
    num_nodes = len(lat_lons)

    torch.manual_seed(42)
    features = torch.randn((batch_size, num_nodes, 78))

    # Original pipeline
    encoder_orig = Encoder(
        lat_lons, resolution=2, input_dim=78, output_dim=256, efficient_batching=False
    )
    processor_orig = Processor(256, num_blocks=9)
    decoder_orig = Decoder(
        lat_lons, resolution=2, input_dim=256, output_dim=78, efficient_batching=False
    )

    encoder_orig.eval()
    processor_orig.eval()
    decoder_orig.eval()

    with torch.no_grad():
        x, edge_idx, edge_attr = encoder_orig(features)
        x = processor_orig(x, edge_idx, edge_attr)
        out_orig = decoder_orig(x, features)

    # Efficient pipeline
    encoder_eff = Encoder(
        lat_lons, resolution=2, input_dim=78, output_dim=256, efficient_batching=True
    )
    processor_eff = Processor(256, num_blocks=9)
    decoder_eff = Decoder(
        lat_lons, resolution=2, input_dim=256, output_dim=78, efficient_batching=True
    )

    encoder_eff.load_state_dict(encoder_orig.state_dict())
    processor_eff.load_state_dict(processor_orig.state_dict())
    decoder_eff.load_state_dict(decoder_orig.state_dict())

    encoder_eff.eval()
    processor_eff.eval()
    decoder_eff.eval()

    with torch.no_grad():
        x_eff, edge_idx_eff, edge_attr_eff = encoder_eff(features)
        x_eff = processor_eff(
            x_eff, edge_idx_eff, edge_attr_eff, batch_size=batch_size, efficient_batching=True
        )
        out_eff = decoder_eff(x_eff, features)

    assert torch.allclose(out_orig, out_eff, atol=1e-4)
