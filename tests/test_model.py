from graph_weather.models import Decoder, Encoder, Processor


def test_encoder():
    lat_lons = []
    for lat in range(-90, 90, 1):
        for lon in range(0, 360, 1):
            lat_lons.append((lat, lon))
    model = Encoder(lat_lons)


def test_decoder():
    lat_lons = []
    for lat in range(-90, 90, 1):
        for lon in range(0, 360, 1):
            lat_lons.append((lat, lon))
    model = Decoder(lat_lons)
