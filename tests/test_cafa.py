import torch
import pytest
from graph_weather.models.cafa.encoder import Encoder
from graph_weather.models.cafa.processor import Processor
from graph_weather.models.cafa.decoder import Decoder
from graph_weather.models.cafa.config import CaFA

def test_encoder():
    """Test that encoder produces correct output shape with downsampling."""
    batch_size = 2
    in_channels = 4
    hidden_dim = 64
    height, width = 32, 32
    downsample_factor = 2
    
    encoder = Encoder(
        in_channels=in_channels, 
        hidden_dim=hidden_dim, 
        downsample_factor=downsample_factor, 
        num_layers=2
    )
    
    x = torch.randn(batch_size, in_channels, height, width)
    output = encoder(x)
    
    # Expected output shape: [B, hidden_dim, H//downsample, W//downsample]
    expected_shape = (batch_size, hidden_dim, height // downsample_factor, width // downsample_factor)
    assert output.shape == expected_shape, f"Encoder output shape {output.shape} != expected {expected_shape}"


def test_processor():
    """Test that processor maintains input shape while applying transformer blocks."""
    batch_size = 2
    channels = 64
    height, width = 16, 16
    depth = 2
    num_heads = 4
    mlp_ratio = 2.0
    
    processor = Processor(
        dim=channels, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_ratio=mlp_ratio
    )
    
    x = torch.randn(batch_size, channels, height, width)
    output = processor(x, height, width)
    
    # Expected output shape is identical to input: [B, C, H, W]
    expected_shape = (batch_size, channels, height, width)
    assert output.shape == expected_shape, f"Processor output shape {output.shape} != expected {expected_shape}"


def test_decoder():
    """Test that decoder produces correct output shape with upsampling."""
    batch_size = 2
    in_channels = 64
    out_channels = 4
    height, width = 16, 16
    upsample_factor = 2
    
    decoder = Decoder(
        in_channels=in_channels, 
        out_channels=out_channels, 
        upsample_factor=upsample_factor, 
        num_layers=2
    )
    
    x = torch.randn(batch_size, in_channels, height, width)
    output = decoder(x)
    
    # Expected output shape: [B, out_channels, H*upsample_factor, W*upsample_factor]
    expected_shape = (batch_size, out_channels, height * upsample_factor, width * upsample_factor)
    assert output.shape == expected_shape, f"Decoder output shape {output.shape} != expected {expected_shape}"


def test_cafa_model():
    """Test the full CaFA model with downsampling and upsampling."""
    batch_size = 2
    in_channels = 4
    hidden_dim = 64
    out_channels = 4
    height, width = 32, 32
    encoder_downsample = 2
    decoder_upsample = 2
    processor_depth = 2
    
    # Instantiate the CaFA model with the given parameters
    model = CaFA(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        encoder_downsample=encoder_downsample,
        decoder_upsample=decoder_upsample,
        processor_depth=processor_depth,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.1,
        encoder_layers=2,
        decoder_layers=2
    )
    
    x = torch.randn(batch_size, in_channels, height, width)
    output = model(x)
    
    # Expected output shape: [B, out_channels, H, W]
    # The encoder downsamples by encoder_downsample factor
    # The decoder upsamples by decoder_upsample factor
    # So final shape should be [B, out_channels, H*decoder_upsample/encoder_downsample, W*decoder_upsample/encoder_downsample]
    expected_height = height * decoder_upsample // encoder_downsample
    expected_width = width * decoder_upsample // encoder_downsample
    expected_shape = (batch_size, out_channels, expected_height, expected_width)
    
    assert output.shape == expected_shape, f"CaFA output shape {output.shape} != expected {expected_shape}"


def test_cafa_full_pipeline():
    """Test the CaFA model with a realistic configuration for weather prediction."""
    batch_size = 4
    in_channels = 10  # Example: multiple weather variables
    hidden_dim = 128
    out_channels = 5  # Example: output predictions for different variables
    height, width = 64, 64  # Grid size
    
    model = CaFA(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        encoder_downsample=2,
        decoder_upsample=2,
        processor_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        encoder_layers=3,
        decoder_layers=3
    )
    
    # Forward pass with random input
    x = torch.randn(batch_size, in_channels, height, width)
    output = model(x)
    
    # Check output dimensions
    expected_shape = (batch_size, out_channels, height, width)
    assert output.shape == expected_shape, f"Full pipeline output shape {output.shape} != expected {expected_shape}"