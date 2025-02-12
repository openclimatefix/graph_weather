import numpy as np
import torch
import pytest
from packaging.version import Version

from torch_geometric.transforms import TwoHop

from graph_weather.models.gencast.utils.noise import (
    generate_isotropic_noise,
    sample_noise_level,
)
from graph_weather.models.gencast import GraphBuilder, WeightedMSELoss, Denoiser, Sampler
from graph_weather.models.gencast.layers.modules import FourierEmbedding


def test_gencast_noise():
    num_lon = 360
    num_lat = 180
    num_samples = 5
    target_residuals = np.zeros((num_lon, num_lat, num_samples))
    noise_level = sample_noise_level()
    noise = generate_isotropic_noise(
        num_lon=num_lon, num_lat=num_lat, num_samples=target_residuals.shape[-1]
    )
    corrupted_residuals = target_residuals + noise_level * noise
    assert corrupted_residuals.shape == target_residuals.shape
    assert not np.isnan(corrupted_residuals).any()

    num_lon = 360
    num_lat = 181
    num_samples = 5
    target_residuals = np.zeros((num_lon, num_lat, num_samples))
    noise_level = sample_noise_level()
    noise = generate_isotropic_noise(
        num_lon=num_lon, num_lat=num_lat, num_samples=target_residuals.shape[-1]
    )
    corrupted_residuals = target_residuals + noise_level * noise
    assert corrupted_residuals.shape == target_residuals.shape
    assert not np.isnan(corrupted_residuals).any()

    num_lon = 100
    num_lat = 100
    num_samples = 5
    target_residuals = np.zeros((num_lon, num_lat, num_samples))
    noise_level = sample_noise_level()
    noise = generate_isotropic_noise(
        num_lon=num_lon, num_lat=num_lat, num_samples=target_residuals.shape[-1], isotropic=False
    )
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


def test_gencast_sampler():
    grid_lat = np.arange(-90, 90, 1)
    grid_lon = np.arange(0, 360, 1)
    input_features_dim = 10
    output_features_dim = 5

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

    prev_inputs = torch.randn((1, len(grid_lon), len(grid_lat), 2 * input_features_dim))

    sampler = Sampler()
    preds = sampler.sample(denoiser, prev_inputs)
    assert not torch.isnan(preds).any()
    assert preds.shape == (1, len(grid_lon), len(grid_lat), output_features_dim)


@pytest.mark.skipif(
    Version(torch.__version__).release != Version("2.3.0").release,
    reason="dgl tests for experimental features only runs with torch 2.3.0",
)
def test_gencast_full():
    # download weights from HF
    denoiser = Denoiser.from_pretrained(
        "openclimatefix/gencast-128x64",
        grid_lon=np.arange(0, 360, 360 / 128),
        grid_lat=np.arange(-90, 90, 180 / 64) + 1 / 2 * 180 / 64,
    )

    # load inputs and targets
    prev_inputs = torch.randn([1, 128, 64, 178])
    target_residuals = torch.randn([1, 128, 64, 83])

    # predict
    sampler = Sampler()
    preds = sampler.sample(denoiser, prev_inputs)

    assert not torch.isnan(preds).any()
    assert preds.shape == target_residuals.shape
