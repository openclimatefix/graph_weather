"""Tests for the stretched-grid regional forecaster."""

import torch

from graph_weather.models.layers.stretched_mesh import (
    build_variable_resolution_mesh,
    mixed_resolution_embedding_indices,
)
from graph_weather.models.stretched_forecast import StretchedForecasterConfig

BBOX = (50.0, 55.0, -2.0, 3.0)
LAT_LONS = [(52.5, 0.5), (53.0, 1.0), (-40.0, 150.0)]


def _small_config():
    """A tiny config so a forward pass runs fast in tests."""
    return StretchedForecasterConfig(
        coarse_res=2,
        fine_res=3,
        feature_dim=4,
        aux_dim=0,
        node_dim=16,
        edge_dim=16,
        num_blocks=1,
        hidden_dim_processor_node=16,
        hidden_dim_processor_edge=16,
        hidden_layers_processor_node=1,
        hidden_layers_processor_edge=1,
        hidden_dim_decoder=16,
        hidden_layers_decoder=1,
    )


def test_forward_returns_one_prediction_per_observation():
    """Forward returns [B, N_obs, output_dim] for a stretched mesh."""
    model = _small_config().build()
    features = torch.randn(1, len(LAT_LONS), 4)

    out = model(features, LAT_LONS, BBOX)

    assert out.shape == (1, len(LAT_LONS), 4)


def test_gradient_reaches_both_embedding_tables():
    """Both the coarse and fine tables receive gradient, so both actually train."""
    model = _small_config().build()
    features = torch.randn(1, len(LAT_LONS), 4)

    # A plain .sum() loss is invariant under the decoder's final LayerNorm (the sum over the
    # normalized dimension is constant), which zeroes every gradient. Use a non-invariant loss.
    model(features, LAT_LONS, BBOX).pow(2).sum().backward()

    assert model.coarse_embeddings.grad is not None
    assert model.fine_embeddings.grad is not None
    assert model.coarse_embeddings.grad.abs().sum() > 0
    assert model.fine_embeddings.grad.abs().sum() > 0


def test_runs_for_a_moved_region():
    """The same model forecasts over a different region without rebuilding."""
    model = _small_config().build()
    japan_bbox = (30.0, 40.0, 135.0, 145.0)
    japan_points = [(35.0, 139.0), (36.0, 140.0), (-40.0, 150.0)]
    features = torch.randn(1, len(japan_points), 4)

    out = model(features, japan_points, japan_bbox)

    assert out.shape == (1, len(japan_points), 4)


def test_handles_batch_dimension():
    """A batch of several samples produces one output per sample."""
    model = _small_config().build()
    features = torch.randn(3, len(LAT_LONS), 4)

    out = model(features, LAT_LONS, BBOX)

    assert out.shape == (3, len(LAT_LONS), 4)


def test_output_dim_defaults_to_feature_dim():
    """With output_dim unset, the model predicts one value per input feature."""
    config = _small_config()
    assert config.output_dim is None
    model = config.build()

    out = model(torch.randn(1, len(LAT_LONS), 4), LAT_LONS, BBOX)

    assert out.shape[-1] == config.feature_dim


def test_cached_rows_match_embedding_index_helper():
    """The model's precomputed row maps reproduce mixed_resolution_embedding_indices."""
    model = _small_config().build()
    mesh = build_variable_resolution_mesh(BBOX, 2, 3)
    expected = mixed_resolution_embedding_indices(mesh, 2, 3)

    for cell, (res, row) in zip(mesh, expected):
        cached = model._fine_rows[cell] if res == 3 else model._coarse_rows[cell]
        assert cached == row


def test_decoder_seeded_with_encoder_obs_features():
    """The decoder's observation nodes carry the encoder's per-obs features, not zeros.

    Seeding them with zeros collapses the model toward persistence, since the predicted delta
    then only sees cell-level context. This guards that regression.
    """
    model = _small_config().build()
    features = torch.randn(1, len(LAT_LONS), 4)

    captured = {}

    def capture(_module, inputs, _output):
        captured["dec_input"] = inputs[0].detach()

    handle = model.decoder_gnn.register_forward_hook(capture)
    model(features, LAT_LONS, BBOX)
    handle.remove()

    obs_rows = captured["dec_input"][: len(LAT_LONS)]
    assert obs_rows.abs().sum() > 0
