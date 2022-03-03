import h3
import torch

from graph_weather import GraphWeatherForecaster
from graph_weather.models import Decoder, Encoder, Processor


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

    features = torch.randn((1, len(lat_lons), 78))

    out = model(features)
    assert not torch.isnan(out).all()

    criterion = torch.nn.MSELoss()
    loss = criterion(out, features)
    loss.backward()
