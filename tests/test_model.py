import h3
import numpy as np
import torch

from graph_weather import GraphWeatherAssimilator, GraphWeatherForecaster
from graph_weather.models import AssimilatorDecoder, AssimilatorEncoder, Decoder, Encoder, Processor
from graph_weather.models.losses import NormalizedMSELoss


def test_encoder():
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))
    model = Encoder(lat_lons).eval()

    features = torch.randn((2, len(lat_lons), 78))
    with torch.no_grad():
        x, edge_idx, edge_attr = model(features)
    assert x.size() == (5882 * 2, 256)
    assert edge_idx.size() == (2, 41162 * 2)


def test_encoder_uneven_grid():
    lat_lons = []
    for lat in range(-90, 90, 7):
        for lon in range(0, 180, 5):
            lat_lons.append((lat, lon))
        for lon in range(180, 360, 9):
            lat_lons.append((lat, lon))
    model = Encoder(lat_lons).eval()

    features = torch.randn((2, len(lat_lons), 78))
    with torch.no_grad():
        x, edge_idx, edge_attr = model(features)
    assert x.size() == (5882 * 2, 256)
    assert edge_idx.size() == (2, 41162 * 2)


def test_assimilation_encoder_uneven_grid():
    lat_lons = []
    for lat in range(-90, 90, 7):
        for lon in range(0, 180, 5):
            lat_lons.append((lat, lon, np.random.random(1)))
        for lon in range(180, 360, 9):
            lat_lons.append((lat, lon, np.random.random(1)))
    model = AssimilatorEncoder().eval()

    features = torch.randn((2, len(lat_lons), 2))
    with torch.no_grad():
        x, edge_idx, edge_attr = model(features, torch.tensor(lat_lons))
    assert x.size() == (5882 * 2, 256)
    assert edge_idx.size() == (2, 41162 * 2)


def test_processor():
    processor = Processor().eval()
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))
    model = Encoder(lat_lons).eval()

    features = torch.randn((3, len(lat_lons), 78))
    with torch.no_grad():
        x, edge_idx, edge_attr = model(features)
        out = processor(x, edge_idx, edge_attr)
    assert out.size() == x.size()


def test_decoder():
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))
    model = Decoder(lat_lons).eval()
    features = torch.randn((3, len(lat_lons), 78))
    processed = torch.randn((3 * h3.num_hexagons(2), 256))
    with torch.no_grad():
        x = model(processed, features)
    assert x.size() == (3, 2592, 78)


def test_assimilator():
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))
    model = AssimilatorDecoder(lat_lons).eval()
    processed = torch.randn((3 * h3.num_hexagons(2), 256))
    with torch.no_grad():
        x = model(processed, 3)
    assert x.size() == (3, 2592, 78)


def test_end2end():
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))
    model = Encoder(lat_lons).eval()
    processor = Processor().eval()
    decoder = Decoder(lat_lons).eval()
    features = torch.randn((4, len(lat_lons), 78))
    with torch.no_grad():
        x, edge_idx, edge_attr = model(features)
        out = processor(x, edge_idx, edge_attr)
        pred = decoder(out, features)
    assert pred.size() == (4, 2592, 78)


def test_forecaster():
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))
    model = GraphWeatherForecaster(lat_lons)
    # Add in auxiliary features
    features = torch.randn((1, len(lat_lons), 78 + 24))

    out = model(features)
    assert not torch.isnan(out).any()
    assert not torch.isnan(out).any()


def test_assimilator_model():
    obs_lat_lons = []
    for lat in range(-90, 90, 7):
        for lon in range(0, 180, 6):
            obs_lat_lons.append((lat, lon, np.random.random(1)))
        for lon in 360 * np.random.random(100):
            obs_lat_lons.append((lat, lon, np.random.random(1)))

    output_lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            output_lat_lons.append((lat, lon))
    model = GraphWeatherAssimilator(output_lat_lons=output_lat_lons, analysis_dim=24)

    features = torch.randn((1, len(obs_lat_lons), 2))
    lat_lon_heights = torch.tensor(obs_lat_lons)
    out = model(features, lat_lon_heights)
    assert not torch.isnan(out).any()
    assert not torch.isnan(out).any()


def test_forecaster_and_loss():
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))
    criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=torch.randn((78,)))
    model = GraphWeatherForecaster(lat_lons)
    # Add in auxiliary features
    features = torch.randn((2, len(lat_lons), 78 + 24))

    out = model(features)
    loss = criterion(out, torch.rand(out.shape))
    assert not torch.isnan(loss)
    assert not torch.isnan(out).any()
    assert not torch.isnan(out).any()
    loss.backward()


def test_assimilator_model_grad_checkpoint():
    obs_lat_lons = []
    for lat in range(-90, 90, 7):
        for lon in range(0, 180, 6):
            obs_lat_lons.append((lat, lon, np.random.random(1)))
        for lon in 360 * np.random.random(100):
            obs_lat_lons.append((lat, lon, np.random.random(1)))

    output_lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            output_lat_lons.append((lat, lon))
    model = GraphWeatherAssimilator(
        output_lat_lons=output_lat_lons, analysis_dim=24, use_checkpointing=True
    )

    features = torch.randn((1, len(obs_lat_lons), 2))
    lat_lon_heights = torch.tensor(obs_lat_lons)
    out = model(features, lat_lon_heights)
    assert not torch.isnan(out).any()
    assert not torch.isnan(out).any()


def test_forecaster_and_loss_grad_checkpoint():
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))
    criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=torch.randn((78,)))
    model = GraphWeatherForecaster(lat_lons, use_checkpointing=True)
    # Add in auxiliary features
    features = torch.randn((2, len(lat_lons), 78 + 24))

    out = model(features)
    loss = criterion(out, torch.rand(out.shape))
    assert not torch.isnan(loss)
    assert not torch.isnan(out).any()
    assert not torch.isnan(out).any()
    loss.backward()
