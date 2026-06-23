"""Tests for RegionalDataset using a synthetic in-memory grid (no cloud)."""

import numpy as np
import torch
import xarray as xr

from graph_weather.data.regional_dataset import CORE_SURFACE, RegionalDataset
from graph_weather.models.regional_forecast import RegionalForecasterConfig


def _synthetic_ds(n_time=4, n_lat=40, n_lon=80):
    """Build a regular lat/lon dataset with the CORE_SURFACE variables."""
    lat = np.linspace(-80.0, 80.0, n_lat)
    lon = np.linspace(-180.0, 179.0, n_lon)
    rng = np.random.default_rng(0)
    data = {
        v: (("time", "latitude", "longitude"), rng.standard_normal((n_time, n_lat, n_lon)))
        for v in CORE_SURFACE
    }
    return xr.Dataset(data, coords={"latitude": lat, "longitude": lon, "time": np.arange(n_time)})


def _dataset(**kwargs):
    """RegionalDataset over the synthetic grid with test-friendly defaults."""
    return RegionalDataset(dataset=_synthetic_ds(), extent_deg=40.0, max_points=100, **kwargs)


def test_sample_shapes():
    """features, target, global_context are [N, F]; lat_lons has N entries."""
    features, lat_lons, target, global_context = _dataset()[0]
    n = features.shape[0]
    assert features.shape == (n, len(CORE_SURFACE))
    assert target.shape == (n, len(CORE_SURFACE))
    assert global_context.shape == (n, len(CORE_SURFACE))
    assert len(lat_lons) == n


def test_len_is_time_minus_one():
    """One sample per t -> t+1 pair."""
    assert len(_dataset()) == 3


def test_lat_lons_is_list_of_tuples():
    """lat_lons must be a plain Python list of (lat, lon) for the graph builder."""
    _, lat_lons, _, _ = _dataset()[0]
    assert isinstance(lat_lons, list)
    assert isinstance(lat_lons[0], tuple) and len(lat_lons[0]) == 2


def test_feeds_regional_forecaster():
    """Loader output drives RegionalForecaster with aux_dim=0 and no shape error."""
    features, lat_lons, _, _ = _dataset()[0]
    model = RegionalForecasterConfig(
        feature_dim=len(CORE_SURFACE), aux_dim=0, node_dim=32, edge_dim=32, num_blocks=2
    ).build()
    out = model(features.unsqueeze(0), lat_lons)
    assert out.shape == (1, features.shape[0], len(CORE_SURFACE))


def test_movable_centers():
    """Different idx samples a different bounding box location."""
    ds = _dataset()
    _, ll0, _, _ = ds[0]
    _, ll1, _, _ = ds[1]
    assert ll0 != ll1


def test_points_in_domain():
    """Sampled coordinates stay within the grid extent for every sample."""
    ds = _dataset()
    for i in range(len(ds)):
        _, lat_lons, _, _ = ds[i]
        lats = [a for a, _ in lat_lons]
        lons = [b for _, b in lat_lons]
        assert min(lats) >= -80.0 and max(lats) <= 80.0
        assert min(lons) >= -180.0 and max(lons) <= 179.0


def test_max_points_cap():
    """Number of points never exceeds max_points."""
    features, _, _, _ = RegionalDataset(dataset=_synthetic_ds(), extent_deg=60.0, max_points=20)[0]
    assert features.shape[0] <= 20


def test_nan_filled():
    """A fully-NaN variable produces no NaN in the output (fill after normalize)."""
    ds_syn = _synthetic_ds()
    ds_syn["total_cloud_cover"][:] = np.nan
    features, _, target, global_context = RegionalDataset(
        dataset=ds_syn, extent_deg=40.0, max_points=100
    )[0]
    assert not torch.isnan(features).any()
    assert not torch.isnan(target).any()
    assert not torch.isnan(global_context).any()


def test_global_context_is_coarser():
    """global_context differs from features (blurring did something)."""
    features, _, _, global_context = _dataset(global_coarsen=8)[0]
    assert global_context.shape == features.shape
    assert not torch.equal(global_context, features)


def test_coarsen_one_is_identity():
    """global_coarsen=1 leaves the field unchanged, so context == features."""
    features, _, _, global_context = _dataset(global_coarsen=1)[0]
    assert torch.allclose(global_context, features)


def test_global_context_feeds_nudging():
    """global_context drives RegionalForecaster with nudging enabled, no shape error."""
    features, lat_lons, _, global_context = _dataset()[0]
    model = RegionalForecasterConfig(
        feature_dim=len(CORE_SURFACE),
        aux_dim=0,
        node_dim=32,
        edge_dim=32,
        num_blocks=2,
        enable_nudging=True,
    ).build()
    out = model(features.unsqueeze(0), lat_lons, global_context=global_context.unsqueeze(0))
    assert out.shape == (1, features.shape[0], len(CORE_SURFACE))
