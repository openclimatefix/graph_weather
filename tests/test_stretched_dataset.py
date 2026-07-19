"""Tests for StretchedDataset using a synthetic in-memory grid (no cloud)."""

import h3
import numpy as np
import torch
import xarray as xr

from graph_weather.data.regional_dataset import CORE_SURFACE
from graph_weather.data.stretched_dataset import StretchedDataset
from graph_weather.models.layers.stretched_mesh import (
    assign_points_to_mesh,
    build_variable_resolution_mesh,
)

COARSE_RES = 2
FINE_RES = 3


def _synthetic_ds(n_time=4, n_lat=37, n_lon=72):
    """Build a regular lat/lon dataset with the CORE_SURFACE variables."""
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon = np.linspace(-180.0, 175.0, n_lon)
    rng = np.random.default_rng(0)
    data = {
        v: (("time", "latitude", "longitude"), rng.standard_normal((n_time, n_lat, n_lon)))
        for v in CORE_SURFACE
    }
    return xr.Dataset(data, coords={"latitude": lat, "longitude": lon, "time": np.arange(n_time)})


def _dataset(**kwargs):
    """StretchedDataset over the synthetic grid with test-friendly defaults."""
    return StretchedDataset(
        dataset=_synthetic_ds(),
        coarse_res=COARSE_RES,
        fine_res=FINE_RES,
        extent_deg=40.0,
        max_regional_points=50,
        **kwargs,
    )


def test_getitem_returns_four_items():
    """__getitem__ yields (features, lat_lons, target, bbox)."""
    result = _dataset()[0]
    assert len(result) == 4


def test_features_shape_matches_lat_lons():
    """features.shape[0] == len(lat_lons) and feature dim == len(variables)."""
    features, lat_lons, target, _ = _dataset()[0]
    assert features.shape[0] == len(lat_lons)
    assert features.shape[1] == len(CORE_SURFACE)
    assert target.shape == features.shape


def test_lat_lons_is_list_of_tuples():
    """lat_lons must be a plain Python list of (lat, lon) for the graph builder."""
    _, lat_lons, _, _ = _dataset()[0]
    assert isinstance(lat_lons, list)
    assert isinstance(lat_lons[0], tuple) and len(lat_lons[0]) == 2


def test_bbox_is_explicit_tuple():
    """The bbox is a 4-tuple of floats, not derived from lat_lons."""
    _, lat_lons, _, bbox = _dataset()[0]
    assert isinstance(bbox, tuple) and len(bbox) == 4
    assert all(isinstance(x, float) for x in bbox)
    # bbox must NOT span the whole globe (it's a regional box, not min/max of all points).
    lat_min, lat_max, _, _ = bbox
    assert lat_max - lat_min < 90.0


def test_len_is_time_minus_one():
    """One sample per t -> t+1 pair."""
    assert len(_dataset()) == 3


def test_global_points_cover_coarse_cells():
    """Every non-refined coarse cell has at least one observation assigned to it."""
    features, lat_lons, _, bbox = _dataset()[0]
    mesh = build_variable_resolution_mesh(bbox, COARSE_RES, FINE_RES)
    assigned = assign_points_to_mesh(lat_lons, mesh, COARSE_RES, FINE_RES)

    coarse_in_mesh = {c for c in mesh if h3.get_resolution(c) == COARSE_RES}
    coarse_with_obs = {c for c in assigned if c in coarse_in_mesh}
    assert coarse_with_obs == coarse_in_mesh


def test_regional_points_inside_bbox():
    """Regional lat_lons are within the returned bbox."""
    _, lat_lons, _, bbox = _dataset()[0]
    lat_min, lat_max, lon_min, lon_max = bbox
    mesh = build_variable_resolution_mesh(bbox, COARSE_RES, FINE_RES)
    assigned = assign_points_to_mesh(lat_lons, mesh, COARSE_RES, FINE_RES)

    for (lat, lon), cell in zip(lat_lons, assigned):
        if h3.get_resolution(cell) == FINE_RES:
            assert lat_min <= lat <= lat_max
            assert lon_min <= lon <= lon_max


def test_no_nan_in_features():
    """A fully-NaN variable produces no NaN in the output (fill after normalize)."""
    ds_syn = _synthetic_ds()
    ds_syn["total_cloud_cover"][:] = np.nan
    features, _, target, _ = StretchedDataset(
        dataset=ds_syn,
        coarse_res=COARSE_RES,
        fine_res=FINE_RES,
        extent_deg=40.0,
        max_regional_points=50,
    )[0]
    assert not torch.isnan(features).any()
    assert not torch.isnan(target).any()


def test_different_idx_moves_region():
    """Different idx samples a different bounding box location."""
    ds = _dataset()
    _, _, _, bbox0 = ds[0]
    _, _, _, bbox1 = ds[1]
    assert bbox0 != bbox1


def test_target_is_next_timestep():
    """Target comes from t+1, not t, so features and target differ."""
    features, _, target, _ = _dataset()[0]
    assert not torch.equal(features, target)


def test_more_points_than_regional_alone():
    """Total points exceed max_regional_points because global points are added."""
    features, _, _, _ = _dataset()[0]
    assert features.shape[0] > 50


def test_global_lat_lons_are_cell_centres():
    """Global observations sit at coarse cell centres, not at IFS grid points."""
    _, lat_lons, _, bbox = _dataset()[0]
    mesh = build_variable_resolution_mesh(bbox, COARSE_RES, FINE_RES)
    # Global points come first in the concatenation; count them from the mesh.
    n_global = sum(1 for c in mesh if h3.get_resolution(c) == COARSE_RES)
    global_lls = lat_lons[:n_global]

    coarse_cells = sorted(h3.uncompact_cells(h3.get_res0_cells(), COARSE_RES))
    mesh_set = set(mesh)
    kept_centres = [
        h3.cell_to_latlng(c) for c in coarse_cells if c in mesh_set
    ]
    for (lat, lon), (clat, clon) in zip(global_lls, kept_centres):
        assert abs(lat - clat) < 1e-6
        assert abs(lon - clon) < 1e-6
