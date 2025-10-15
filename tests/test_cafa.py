import torch
import pytest

from graph_weather.models.cafa.encoder import CaFAEncoder
from graph_weather.models.cafa.processor import CaFAProcessor
from graph_weather.models.cafa.decoder import CaFADecoder
from graph_weather.models.cafa.model import CaFAForecaster

#common params for test
BATCH_SIZE = 2
HEIGHT = 32
WIDTH = 64
MODEL_DIM = 128
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3
HEADS = 4
DEPTH = 2
DOWNSAMPLING = 2

def test_encoder():
    """Tests the CaFAEncoder for correct shape transformation."""
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)
    encoder = CaFAEncoder(
        input_channels=INPUT_CHANNELS,
        model_dim=MODEL_DIM,
        downsampling_factor=DOWNSAMPLING
    )
    output = encoder(x)
    
    assert output.shape == (BATCH_SIZE, MODEL_DIM, HEIGHT // DOWNSAMPLING, WIDTH // DOWNSAMPLING)

def test_decoder():
    """Tests the CaFADecoder for correct shape transformation."""
    x = torch.randn(BATCH_SIZE, MODEL_DIM, HEIGHT // DOWNSAMPLING, WIDTH // DOWNSAMPLING)
    decoder = CaFADecoder(
        model_dim=MODEL_DIM,
        output_channels=OUTPUT_CHANNELS,
        upsampling_factor=DOWNSAMPLING
    )
    output = decoder(x)
    
    assert output.shape == (BATCH_SIZE, OUTPUT_CHANNELS, HEIGHT, WIDTH)

def test_processor():
    """Tests the CaFAProcessor to ensure it preserves shape."""
    x = torch.randn(BATCH_SIZE, MODEL_DIM, HEIGHT, WIDTH)
    processor = CaFAProcessor(dim=MODEL_DIM, depth=DEPTH, heads=HEADS)
    output = processor(x)
    
    assert output.shape == x.shape

def test_cafa_forecaster_end_to_end():
    """Tests the full CaFAForecaster model to ensure it works end-to-end."""
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)
    model = CaFAForecaster(
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        model_dim=MODEL_DIM,
        downsampling_factor=DOWNSAMPLING,
        processor_depth=DEPTH,
        num_heads=HEADS
    )
    output = model(x)
    
    assert output.shape == (BATCH_SIZE, OUTPUT_CHANNELS, HEIGHT, WIDTH)

def test_cafa_forecaster_odd_dimensions():
    """Tests that the model's internal padding handles odd-sized inputs correctly."""
    
    # Use odd dimensions for height and width
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT + 1, WIDTH + 1)
    model = CaFAForecaster(
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        model_dim=MODEL_DIM,
        downsampling_factor=DOWNSAMPLING,
        processor_depth=DEPTH,
        num_heads=HEADS
    )
    output = model(x)
    
    # The model should return a tensor with the original odd-sized dimensions
    assert output.shape == x.shape