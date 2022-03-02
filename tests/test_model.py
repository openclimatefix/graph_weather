import h3
import torch

from graph_weather.models import Decoder, Encoder, Processor


def test_encoder():
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))
    model = Encoder(lat_lons).eval()

    features = torch.randn((len(lat_lons), 78))
    with torch.no_grad():
        x, edge_idx, edge_attr = model(features)
    assert x.size() == (5882, 256)
    assert edge_idx.size() == (2, 41162)


def test_processor():
    processor = Processor().eval()
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))
    model = Encoder(lat_lons).eval()

    features = torch.randn((len(lat_lons), 78))
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
    features = torch.randn((len(lat_lons), 78))
    processed = torch.randn((h3.num_hexagons(2), 256))
    with torch.no_grad():
        x = model(processed, features)
        print(x.shape)


test_decoder()
