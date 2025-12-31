"""
Test script to verify that efficient batching produces the same outputs as original batching.
"""

import numpy as np
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


def test_encoder_outputs(resolution_deg=5.0, batch_size=2):
    """Test that Encoder produces identical outputs with both batching modes."""
    print(f"\n{'='*80}")
    print(f"Testing Encoder with {resolution_deg}° grid, batch_size={batch_size}")
    print(f"{'='*80}\n")

    lat_lons = create_lat_lon_grid(resolution_deg)
    num_nodes = len(lat_lons)

    # Create input
    torch.manual_seed(42)
    features = torch.randn((batch_size, num_nodes, 78))

    # Original batching
    print("Running original batching...")
    encoder_original = Encoder(
        lat_lons, resolution=2, input_dim=78, output_dim=256, efficient_batching=False
    )
    encoder_original.eval()

    with torch.no_grad():
        x_orig, edge_idx_orig, edge_attr_orig = encoder_original(features)

    print(f"  Output shape: {x_orig.shape}")
    print(f"  Edge index shape: {edge_idx_orig.shape}")
    print(f"  Edge attr shape: {edge_attr_orig.shape}")

    # Efficient batching
    print("\nRunning efficient batching...")
    encoder_efficient = Encoder(
        lat_lons, resolution=2, input_dim=78, output_dim=256, efficient_batching=True
    )
    encoder_efficient.load_state_dict(encoder_original.state_dict())  # Same weights
    encoder_efficient.eval()

    with torch.no_grad():
        x_eff, edge_idx_eff, edge_attr_eff = encoder_efficient(features)

    print(f"  Output shape: {x_eff.shape}")
    print(f"  Edge index shape: {edge_idx_eff.shape}")
    print(f"  Edge attr shape: {edge_attr_eff.shape}")

    # Compare outputs
    print("\nComparing outputs...")
    print(f"  Node features - Max diff: {torch.max(torch.abs(x_orig - x_eff)).item():.2e}")
    print(f"  Node features - Mean diff: {torch.mean(torch.abs(x_orig - x_eff)).item():.2e}")

    # Edge structures will differ: batched (original) vs shared (efficient)
    print("\n  Edge structure comparison:")
    print(f"    Original (batched): {edge_idx_orig.shape[1]} edges")
    print(f"    Efficient (shared): {edge_idx_eff.shape[1]} edges")
    print(
        f"    Reduction: {edge_idx_orig.shape[1] / edge_idx_eff.shape[1]:.1f}x fewer edges with shared graph!"
    )

    # For deep optimization, edge structures are intentionally different
    # Original: replicated B times, Efficient: shared single copy
    # We can't compare them directly, but we verify the reduction
    expected_reduction = batch_size
    actual_reduction = edge_idx_orig.shape[1] / edge_idx_eff.shape[1]
    edges_correct = abs(actual_reduction - expected_reduction) < 0.1

    print(f"    Expected reduction: {expected_reduction}x")
    print(f"    Actual reduction: {actual_reduction:.1f}x")
    print(f"    Edge structure: {'✓ Correct' if edges_correct else '✗ Incorrect'}")

    # Check if node features match (this is what matters for correctness)
    outputs_match = torch.allclose(x_orig, x_eff, atol=1e-5) and edges_correct

    if outputs_match:
        print("\n✓ SUCCESS: Node features match and graph sharing is correct!")
    else:
        print("\n✗ FAILED: Node features differ or graph structure incorrect!")

    return outputs_match


def test_decoder_outputs(resolution_deg=5.0, batch_size=2):
    """Test that Decoder produces identical outputs with both batching modes."""
    print(f"\n{'='*80}")
    print(f"Testing Decoder with {resolution_deg}° grid, batch_size={batch_size}")
    print(f"{'='*80}\n")

    lat_lons = create_lat_lon_grid(resolution_deg)
    num_nodes = len(lat_lons)

    # Create input (processor features)
    torch.manual_seed(42)
    num_h3 = 5882  # Standard H3 resolution 2
    processor_features = torch.randn((batch_size * num_h3, 256))
    start_features = torch.randn((batch_size, num_nodes, 78))

    # Original batching
    print("Running original batching...")
    decoder_original = Decoder(
        lat_lons, resolution=2, input_dim=256, output_dim=78, efficient_batching=False
    )
    decoder_original.eval()

    with torch.no_grad():
        out_orig = decoder_original(processor_features, start_features)

    print(f"  Output shape: {out_orig.shape}")

    # Efficient batching
    print("\nRunning efficient batching...")
    decoder_efficient = Decoder(
        lat_lons, resolution=2, input_dim=256, output_dim=78, efficient_batching=True
    )
    decoder_efficient.load_state_dict(decoder_original.state_dict())  # Same weights
    decoder_efficient.eval()

    with torch.no_grad():
        out_eff = decoder_efficient(processor_features, start_features)

    print(f"  Output shape: {out_eff.shape}")

    # Compare outputs
    print("\nComparing outputs...")
    print(f"  Max diff: {torch.max(torch.abs(out_orig - out_eff)).item():.2e}")
    print(f"  Mean diff: {torch.mean(torch.abs(out_orig - out_eff)).item():.2e}")

    outputs_match = torch.allclose(out_orig, out_eff, atol=1e-5)

    if outputs_match:
        print("\n✓ SUCCESS: Decoder outputs match between original and efficient batching!")
    else:
        print("\n✗ FAILED: Decoder outputs differ!")

    return outputs_match


def test_full_pipeline(resolution_deg=5.0, batch_size=2):
    """Test the full encoder->processor->decoder pipeline."""
    print(f"\n{'='*80}")
    print(f"Testing Full Pipeline with {resolution_deg}° grid, batch_size={batch_size}")
    print(f"{'='*80}\n")

    lat_lons = create_lat_lon_grid(resolution_deg)
    num_nodes = len(lat_lons)

    # Create input
    torch.manual_seed(42)
    features = torch.randn((batch_size, num_nodes, 78))

    # Original pipeline
    print("Running original pipeline...")
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

    print(f"  Final output shape: {out_orig.shape}")
    print(f"  Edge index shape (batched): {edge_idx.shape}")

    # Efficient pipeline with DEEP optimization
    print("\nRunning efficient pipeline (DEEP optimization)...")
    encoder_eff = Encoder(
        lat_lons, resolution=2, input_dim=78, output_dim=256, efficient_batching=True
    )
    processor_eff = Processor(256, num_blocks=9)
    decoder_eff = Decoder(
        lat_lons, resolution=2, input_dim=256, output_dim=78, efficient_batching=True
    )

    # Load same weights
    encoder_eff.load_state_dict(encoder_orig.state_dict())
    processor_eff.load_state_dict(processor_orig.state_dict())
    decoder_eff.load_state_dict(decoder_orig.state_dict())

    encoder_eff.eval()
    processor_eff.eval()
    decoder_eff.eval()

    with torch.no_grad():
        x_eff, edge_idx_eff, edge_attr_eff = encoder_eff(features)
        print(f"  Edge index shape (shared): {edge_idx_eff.shape}")
        print(
            f"  → Graph replication avoided: {edge_idx.shape[1] // edge_idx_eff.shape[1]}x fewer edges!"
        )

        x_eff = processor_eff(
            x_eff, edge_idx_eff, edge_attr_eff, batch_size=batch_size, efficient_batching=True
        )
        out_eff = decoder_eff(x_eff, features)

    print(f"  Final output shape: {out_eff.shape}")

    # Compare
    print("\nComparing final outputs...")
    print(f"  Max diff: {torch.max(torch.abs(out_orig - out_eff)).item():.2e}")
    print(f"  Mean diff: {torch.mean(torch.abs(out_orig - out_eff)).item():.2e}")

    outputs_match = torch.allclose(out_orig, out_eff, atol=1e-4)

    if outputs_match:
        print("\n✓ SUCCESS: Full pipeline outputs match!")
    else:
        print("\n✗ FAILED: Full pipeline outputs differ!")

    return outputs_match


if __name__ == "__main__":
    print("=" * 80)
    print("EFFICIENT BATCHING VALIDATION TESTS")
    print("=" * 80)

    # Test individual components
    encoder_ok = test_encoder_outputs(resolution_deg=5.0, batch_size=2)
    decoder_ok = test_decoder_outputs(resolution_deg=5.0, batch_size=2)
    full_ok = test_full_pipeline(resolution_deg=5.0, batch_size=2)

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY - DEEP OPTIMIZATION")
    print(f"{'='*80}")
    print(f"  Encoder (with graph sharing): {'PASS ✓' if encoder_ok else 'FAIL ✗'}")
    print(f"  Decoder: {'PASS ✓' if decoder_ok else 'FAIL ✗'}")
    print(f"  Full Pipeline: {'PASS ✓' if full_ok else 'FAIL ✗'}")

    if all([encoder_ok, decoder_ok, full_ok]):
        print("\n✓ All tests passed! Deep optimization is working correctly.")
        print("  → Latent graph is now SHARED instead of replicated")
        print("  → Node features remain identical (no accuracy loss)")
        print("  → Ready for benchmarking!")
    else:
        print("\n✗ Some tests failed. Check implementation.")
