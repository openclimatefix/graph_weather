"""Tests for the variable-resolution (stretched) H3 mesh builder."""

import h3
import pytest

from graph_weather.models.layers.stretched_mesh import (
    assign_points_to_mesh,
    build_variable_resolution_mesh,
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


def test_assignment_one_cell_per_point():
    """Each input point gets exactly one assigned cell, order preserved."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)
    points = [(52.5, 0.5), (-40.0, 150.0), (0.0, 0.0)]
    assigned = assign_points_to_mesh(points, mesh, coarse_res=2, fine_res=4)
    assert len(assigned) == len(points)


def test_point_in_region_assigned_fine_cell():
    """A point inside the refined region is assigned its fine cell."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)
    assigned = assign_points_to_mesh([(52.5, 0.5)], mesh, coarse_res=2, fine_res=4)
    assert h3.get_resolution(assigned[0]) == 4
    assert assigned[0] == h3.latlng_to_cell(52.5, 0.5, 4)


def test_point_outside_region_assigned_coarse_cell():
    """A point outside the region is assigned its coarse cell."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)
    assigned = assign_points_to_mesh([(-40.0, 150.0)], mesh, coarse_res=2, fine_res=4)
    assert h3.get_resolution(assigned[0]) == 2
    assert assigned[0] == h3.latlng_to_cell(-40.0, 150.0, 2)


def test_every_assigned_cell_is_in_mesh():
    """Every assigned cell is a member of the mesh (fine inside, coarse outside)."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)
    points = [(52.5, 0.5), (53.0, 1.0), (-40.0, 150.0), (0.0, 0.0), (80.0, -100.0)]
    assigned = assign_points_to_mesh(points, mesh, coarse_res=2, fine_res=4)
    mesh_set = set(mesh)
    assert all(cell in mesh_set for cell in assigned)


def test_assigned_cell_contains_its_point():
    """The assigned cell is the H3 cell that actually contains the point."""
    mesh = build_variable_resolution_mesh(BBOX, coarse_res=2, fine_res=4)
    points = [(52.5, 0.5), (-40.0, 150.0), (0.0, 0.0), (80.0, -100.0)]
    assigned = assign_points_to_mesh(points, mesh, coarse_res=2, fine_res=4)
    for (lat, lon), cell in zip(points, assigned):
        assert h3.latlng_to_cell(lat, lon, h3.get_resolution(cell)) == cell
