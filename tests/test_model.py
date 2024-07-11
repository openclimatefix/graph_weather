import h3
import numpy as np
import torch
from torch_geometric.transforms import TwoHop

from graph_weather import GraphWeatherAssimilator, GraphWeatherForecaster
from graph_weather.models import (
    AssimilatorDecoder,
    AssimilatorEncoder,
    Decoder,
    Encoder,
    Processor,
    ImageMetaModel,
    MetaModel,
    WrapperImageModel,
    WrapperMetaModel,
)
from graph_weather.models.losses import NormalizedMSELoss

from graph_weather.models.gencast.utils.noise import (
    generate_isotropic_noise,
    sample_noise_level,
)
from graph_weather.models.gencast import GraphBuilder, WeightedMSELoss, Denoiser
from graph_weather.models.gencast.layers.modules import FourierEmbedding


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


def test_normalized_loss():
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))

    # Generate output as strictly positive random features
    out = torch.rand((2, len(lat_lons), 78)) + 0.0001
    feature_variance = out**2
    target = torch.zeros((2, len(lat_lons), 78))

    criterion = NormalizedMSELoss(
        lat_lons=lat_lons, feature_variance=feature_variance, normalize=True
    )
    loss = criterion(out, target)

    assert not torch.isnan(loss)
    # Since feature_variance = out**2 and target = 0, we expect loss = weights
    assert torch.isclose(loss, criterion.weights.expand_as(out.mean(-1)).mean())


def test_image_meta_model():
    batch = 2
    channels = 3
    size = 4
    patch_size = 2
    image = torch.randn((batch, channels, size, size))
    model = ImageMetaModel(
        image_size=size,
        patch_size=patch_size,
        channels=channels,
        depth=1,
        heads=1,
        mlp_dim=7,
        dim_head=64,
    )

    out = model(image)
    assert not torch.isnan(out).any()
    assert not torch.isnan(out).any()
    assert out.size() == image.size()


def test_wrapper_image_meta_model():
    batch = 2
    channels = 3
    size = 4
    patch_size = 2
    model = ImageMetaModel(
        image_size=size,
        patch_size=patch_size,
        channels=channels,
        depth=1,
        heads=1,
        mlp_dim=7,
        dim_head=64,
    )
    scale_factor = 3
    big_image = torch.randn((batch, channels, size * scale_factor, size * scale_factor))
    big_model = WrapperImageModel(model, scale_factor)
    out = big_model(big_image)
    assert not torch.isnan(out).any()
    assert not torch.isnan(out).any()
    assert out.size() == big_image.size()


def test_meta_model():
    lat_lons = []
    for lat in range(-90, 90, 5):
        for lon in range(0, 360, 5):
            lat_lons.append((lat, lon))

    batch = 2
    channels = 3
    image_size = 20
    patch_size = 4
    model = MetaModel(
        lat_lons,
        image_size=image_size,
        patch_size=patch_size,
        depth=1,
        heads=1,
        mlp_dim=7,
        channels=channels,
        dim_head=64,
    )
    features = torch.randn((batch, len(lat_lons), channels))

    out = model(features)
    assert not torch.isnan(out).any()
    assert not torch.isnan(out).any()
    assert out.size() == features.size()


def test_gencast_noise():
    num_lat = 32
    num_samples = 5
    target_residuals = np.zeros((2 * num_lat, num_lat, num_samples))
    noise_level = sample_noise_level()
    noise = generate_isotropic_noise(num_lat=num_lat, num_samples=target_residuals.shape[-1])
    corrupted_residuals = target_residuals + noise_level * noise
    assert corrupted_residuals.shape == target_residuals.shape
    assert not np.isnan(corrupted_residuals).any()


def test_gencast_graph():
    grid_lat = np.arange(-90, 90, 1)
    grid_lon = np.arange(0, 360, 1)
    graphs = GraphBuilder(grid_lon=grid_lon, grid_lat=grid_lat, splits=4, num_hops=8)

    # compare khop sparse implementation with pyg.
    transform = TwoHop()
    khop_mesh_graph_pyg = graphs.mesh_graph
    for i in range(3):  # 8-hop mesh
        khop_mesh_graph_pyg = transform(khop_mesh_graph_pyg)

    assert graphs.mesh_graph.x.shape[0] == 2562
    assert graphs.g2m_graph["grid_nodes"].x.shape[0] == 360 * 180
    assert graphs.m2g_graph["mesh_nodes"].x.shape[0] == 2562
    assert not torch.isnan(graphs.mesh_graph.edge_attr).any()
    assert graphs.khop_mesh_graph.x.shape[0] == 2562
    assert torch.allclose(graphs.khop_mesh_graph.x, khop_mesh_graph_pyg.x)
    assert torch.allclose(graphs.khop_mesh_graph.edge_index, khop_mesh_graph_pyg.edge_index)


def test_gencast_loss():
    grid_lat = torch.arange(-90, 90, 1)
    grid_lon = torch.arange(0, 360, 1)
    pressure_levels = torch.tensor(
        [50.0, 100.0, 150.0, 200.0, 250, 300, 400, 500, 600, 700, 850, 925, 1000.0]
    )
    single_features_weights = torch.tensor([1, 0.1, 0.1, 0.1, 0.1])
    num_atmospheric_features = 6
    batch_size = 3
    features_dim = len(pressure_levels) * num_atmospheric_features + len(single_features_weights)

    loss = WeightedMSELoss(
        grid_lat=grid_lat,
        pressure_levels=pressure_levels,
        num_atmospheric_features=num_atmospheric_features,
        single_features_weights=single_features_weights,
    )

    preds = torch.rand((batch_size, len(grid_lon), len(grid_lat), features_dim))
    noise_levels = torch.rand((batch_size, 1))
    targets = torch.rand((batch_size, len(grid_lon), len(grid_lat), features_dim))
    assert loss.forward(preds, noise_levels, targets) is not None


def test_gencast_denoiser():
    grid_lat = np.arange(-90, 90, 1)
    grid_lon = np.arange(0, 360, 1)
    input_features_dim = 10
    output_features_dim = 5
    batch_size = 3

    denoiser = Denoiser(
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        input_features_dim=input_features_dim,
        output_features_dim=output_features_dim,
        hidden_dims=[16, 32],
        num_blocks=3,
        num_heads=4,
        splits=0,
        num_hops=1,
        device=torch.device("cpu"),
    ).eval()

    corrupted_targets = torch.randn((batch_size, len(grid_lon), len(grid_lat), output_features_dim))
    prev_inputs = torch.randn((batch_size, len(grid_lon), len(grid_lat), 2 * input_features_dim))
    noise_levels = torch.rand((batch_size, 1))

    with torch.no_grad():
        preds = denoiser(
            corrupted_targets=corrupted_targets, prev_inputs=prev_inputs, noise_levels=noise_levels
        )

    assert not torch.isnan(preds).any()


def test_gencast_fourier():
    batch_size = 10
    output_dim = 20
    fourier_embedder = FourierEmbedding(output_dim=output_dim, num_frequencies=32, base_period=16)
    t = torch.rand((batch_size, 1))
    assert fourier_embedder(t).shape == (batch_size, output_dim)
