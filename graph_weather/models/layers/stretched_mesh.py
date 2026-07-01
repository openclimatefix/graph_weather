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


def _global_row_map(res: int) -> dict[str, int]:
    """Map every H3 cell at ``res`` to its row in the sorted global cell set."""
    return {cell: i for i, cell in enumerate(sorted(h3.uncompact_cells(h3.get_res0_cells(), res)))}


def mixed_resolution_embedding_indices(
    cells: list[str],
    coarse_res: int,
    fine_res: int,
) -> list[tuple[int, int]]:
    """Map each mesh cell to its row in a global, per-resolution embedding table.

    A stretched mesh mixes coarse and fine cells, so one table indexed by a single
    resolution no longer fits. Each cell is placed in the table for its own resolution, at
    the row given by its position in ``sorted(uncompact_cells(get_res0_cells(), res))`` - the
    same global numbering ``DynamicGraphBuilder`` uses. Because the row depends only on the
    cell, a location keeps its learned vector as the refined region moves rather than
    relearning it.

    Args:
        cells: H3 cell indices of the mesh, e.g. from ``build_variable_resolution_mesh``.
        coarse_res: H3 resolution used outside the refined region.
        fine_res: H3 resolution used inside the refined region.

    Returns:
        One ``(resolution, row)`` per input cell, in the same order. ``row`` indexes the
        global table for that cell's resolution.
    """
    coarse_rows = _global_row_map(coarse_res)
    fine_rows = _global_row_map(fine_res)

    indices = []
    for cell in cells:
        res = h3.get_resolution(cell)
        row = fine_rows[cell] if res == fine_res else coarse_rows[cell]
        indices.append((res, row))
    return indices
