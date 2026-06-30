"""Build a variable-resolution ("stretched") H3 mesh.

The mesh is coarse over the whole globe and fine over a chosen bounding box, with no
explicit boundary between the two regions. This is the single-mesh alternative to the
dual-mesh-plus-nudging approach: one set of H3 cells whose resolution varies by location.
"""

import h3


def build_variable_resolution_mesh(
    bbox: tuple[float, float, float, float],
    coarse_res: int,
    fine_res: int,
) -> list[str]:
    """Build a globe-covering H3 mesh that is refined over a bounding box.

    The globe is tiled at ``coarse_res``. Every coarse cell whose centre falls inside
    ``bbox`` is replaced by its descendants at ``fine_res``. The result tiles the globe
    exactly once: it is gap-free and has no overlapping (parent + child) cells.

    Because cell selection uses centroid containment, ``coarse_res`` must be fine enough
    that at least one coarse cell centre lands inside ``bbox``; otherwise no refinement
    happens and the mesh is uniformly coarse.

    Args:
        bbox: Region to refine as ``(lat_min, lat_max, lon_min, lon_max)`` in degrees.
        coarse_res: H3 resolution used outside the region.
        fine_res: H3 resolution used inside the region. Must be greater than ``coarse_res``.

    Returns:
        H3 cell indices (string form) for the mixed-resolution mesh, sorted for
        reproducibility.

    Raises:
        ValueError: If ``fine_res`` is not greater than ``coarse_res``.
    """
    if fine_res <= coarse_res:
        raise ValueError(f"fine_res ({fine_res}) must be greater than coarse_res ({coarse_res})")

    lat_min, lat_max, lon_min, lon_max = bbox
    region_polygon = h3.LatLngPoly(
        [(lat_min, lon_min), (lat_max, lon_min), (lat_max, lon_max), (lat_min, lon_max)]
    )
    region_coarse = set(h3.polygon_to_cells(region_polygon, coarse_res))

    coarse_globe = set(h3.uncompact_cells(h3.get_res0_cells(), coarse_res))

    fine_cells: set[str] = set()
    for cell in region_coarse:
        fine_cells.update(h3.cell_to_children(cell, fine_res))

    mesh = (coarse_globe - region_coarse) | fine_cells
    return sorted(mesh)


def assign_points_to_mesh(
    lat_lons: list[tuple[float, float]],
    mesh: list[str],
    coarse_res: int,
    fine_res: int,
) -> list[str]:
    """Assign each lat/lon point to the cell it belongs to in a variable-resolution mesh.

    A point inside the refined region lands on its fine cell; a point outside lands on its
    coarse cell. The fine cell is tried first and used when it is present in ``mesh``,
    otherwise the point's coarse cell is used. Because the mesh refines whole coarse cells
    into all of their children, the fine cell is present exactly when the point sits in the
    region, so every returned cell is a member of ``mesh`` and contains its point.

    Args:
        lat_lons: Observation points as ``(lat, lon)`` in degrees.
        mesh: H3 cell indices of the mesh, e.g. from ``build_variable_resolution_mesh``.
        coarse_res: H3 resolution used outside the refined region.
        fine_res: H3 resolution used inside the refined region.

    Returns:
        One H3 cell index per input point, in the same order.
    """
    mesh_set = set(mesh)
    assigned = []
    for lat, lon in lat_lons:
        fine = h3.latlng_to_cell(lat, lon, fine_res)
        assigned.append(fine if fine in mesh_set else h3.latlng_to_cell(lat, lon, coarse_res))
    return assigned
