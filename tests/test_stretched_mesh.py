"""Tests for the variable-resolution (stretched) H3 mesh builder."""

import h3
import pytest

from graph_weather.models.layers.stretched_mesh import (
    build_variable_resolution_mesh,
    mixed_resolution_embedding_indices,
)

# Bounding box (lat_min, lat_max, lon_min, lon_max) over the UK, large enough that
# polygon_to_cells finds region cells at coarse_res=2.
BBOX = (50.0, 55.0, -2.0, 3.0)


def test_mesh_size_and_no_duplicates():
    """Mesh replaces region coarse cells with their fine children, no duplicates."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)

    # globe@res2 = 5882, region@res2 = 2, each res2 cell has 49 res4 children:
    # 5882 - 2 + 2 * 49 = 5978.
    assert len(mesh) == 5978
    assert len(set(mesh)) == 5978


def test_only_two_resolutions_present():
    """Every cell is at either the coarse or the fine resolution."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)

    resolutions = {h3.get_resolution(cell) for cell in mesh}
    assert resolutions == {2, 4}


def test_region_refined_to_fine():
    """A point inside the bbox lands in a fine cell that is in the mesh."""
    mesh = set(build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4))

    inside_fine = h3.latlng_to_cell(52.5, 0.5, 4)
    assert inside_fine in mesh

    # The coarse cells covering the region must not appear (no parent + child overlap).
    region_coarse = h3.polygon_to_cells(h3.LatLngPoly([(50, -2), (55, -2), (55, 3), (50, 3)]), 2)
    assert all(cell not in mesh for cell in region_coarse)


def test_outside_region_stays_coarse():
    """A point far from the bbox lands in a coarse cell that is in the mesh."""
    mesh = set(build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4))

    outside_coarse = h3.latlng_to_cell(-40.0, 150.0, 2)
    assert outside_coarse in mesh


def test_exact_global_coverage():
    """Mesh tiles the whole globe exactly once when expanded to the fine resolution."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)

    mesh_fine = list(h3.uncompact_cells(mesh, 4))
    globe_fine = list(h3.uncompact_cells(h3.uncompact_cells(h3.get_res0_cells(), 2), 4))

    assert len(mesh_fine) == len(set(mesh_fine))  # no overlaps
    assert set(mesh_fine) == set(globe_fine)  # complete, gap-free coverage


def test_returns_sorted_for_reproducibility():
    """Cells come back in sorted order so the mesh is deterministic across runs."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)
    assert mesh == sorted(mesh)


def test_fine_res_must_exceed_coarse_res():
    """A fine resolution that is not finer than the coarse one is rejected."""
    with pytest.raises(ValueError):
        build_variable_resolution_mesh(BBOX, coarse_res=4, fine_res=4)


def test_index_one_entry_per_cell_with_matching_resolution():
    """Each mesh cell gets one (resolution, row) entry, resolution matching the cell."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)
    indices = mixed_resolution_embedding_indices(mesh, coarse_res=2, fine_res=4)

    assert len(indices) == len(mesh)
    for cell, (res, _row) in zip(mesh, indices):
        assert res == h3.get_resolution(cell)


def test_index_row_within_resolution_table():
    """Every row sits inside its own resolution's global table range."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)
    indices = mixed_resolution_embedding_indices(mesh, coarse_res=2, fine_res=4)

    for res, row in indices:
        assert 0 <= row < h3.get_num_cells(res)


def test_index_matches_global_cell_numbering():
    """A row equals the cell's position in the sorted global cell set at its resolution."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)
    indices = mixed_resolution_embedding_indices(mesh, coarse_res=2, fine_res=4)

    coarse_globe = sorted(h3.uncompact_cells(h3.get_res0_cells(), 2))
    fine_globe = sorted(h3.uncompact_cells(h3.get_res0_cells(), 4))
    coarse_map = {c: i for i, c in enumerate(coarse_globe)}
    fine_map = {c: i for i, c in enumerate(fine_globe)}

    for cell, (res, row) in zip(mesh, indices):
        expected = coarse_map[cell] if res == 2 else fine_map[cell]
        assert row == expected


def test_coarse_index_persists_when_region_moves():
    """A coarse cell keeps its row when the refined region moves elsewhere."""
    uk = build_variable_resolution_mesh((50.0, 55.0, -2.0, 3.0), coarse_res=2, fine_res=4)
    japan = build_variable_resolution_mesh((30.0, 40.0, 135.0, 145.0), coarse_res=2, fine_res=4)

    uk_idx = dict(zip(uk, mixed_resolution_embedding_indices(uk, 2, 4)))
    japan_idx = dict(zip(japan, mixed_resolution_embedding_indices(japan, 2, 4)))

    common_coarse = [c for c in set(uk) & set(japan) if h3.get_resolution(c) == 2]
    assert common_coarse
    for cell in common_coarse:
        assert uk_idx[cell] == japan_idx[cell]


def test_fine_index_persists_across_different_regions():
    """A fine cell shared by two meshes keeps the same row in both."""
    small = build_variable_resolution_mesh((50.0, 55.0, -2.0, 3.0), coarse_res=2, fine_res=4)
    large = build_variable_resolution_mesh((48.0, 57.0, -4.0, 5.0), coarse_res=2, fine_res=4)

    fine_cell = h3.latlng_to_cell(52.5, 0.5, 4)
    small_idx = dict(zip(small, mixed_resolution_embedding_indices(small, 2, 4)))
    large_idx = dict(zip(large, mixed_resolution_embedding_indices(large, 2, 4)))

    assert fine_cell in small_idx and fine_cell in large_idx
    assert small_idx[fine_cell] == large_idx[fine_cell]
