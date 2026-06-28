"""Build a message-passing graph over a variable-resolution ("stretched") H3 mesh.

The mesh from ``build_variable_resolution_mesh`` mixes two H3 resolutions: coarse cells
over the globe and fine cells over a region. Same-resolution neighbours are found with
``grid_disk``, but that never crosses the coarse/fine seam, so the fine region would be a
disconnected island. This module adds the missing seam edges: each fine cell is joined to
the coarse cells bordering its coarse ancestor, found through ``cell_to_parent``.
"""

from typing import List

import h3
import numpy as np
import torch
from torch_geometric.data import Data


def build_variable_resolution_latent_graph(cells: List[str]) -> Data:
    """Wire a latent graph over a mixed-resolution H3 cell set.

    Args:
        cells: H3 cell indices from a variable-resolution mesh, mixing a coarse globe
            resolution with a finer region resolution.

    Returns:
        A ``torch_geometric`` ``Data`` with ``edge_index`` [2, E] and ``edge_attr`` [E, 2]
        holding ``[sin(d), cos(d)]`` of the great-circle distance between cell centres.
        Nodes are the cells in sorted order; edges are bidirectional. Each cell links to
        its same-resolution ring neighbours, and every fine cell additionally links to the
        coarse cells adjacent to its coarse ancestor, stitching the seam closed.
    """
    cells = sorted(cells)
    idx = {cell: i for i, cell in enumerate(cells)}
    cell_set = set(cells)
    coarse_res = min(h3.get_resolution(cell) for cell in cells)

    edges = set()
    for cell in cells:
        source = idx[cell]
        for neighbor in h3.grid_disk(cell, 1):
            if neighbor != cell and neighbor in idx:
                edges.add((source, idx[neighbor]))
        if h3.get_resolution(cell) > coarse_res:
            parent = h3.cell_to_parent(cell, coarse_res)
            for coarse in h3.grid_disk(parent, 1):
                if coarse in cell_set and h3.get_resolution(coarse) == coarse_res:
                    edges.add((source, idx[coarse]))
                    edges.add((idx[coarse], source))

    sources, targets, attrs = [], [], []
    for source, target in sorted(edges):
        dist = h3.great_circle_distance(
            h3.cell_to_latlng(cells[source]), h3.cell_to_latlng(cells[target]), unit="rads"
        )
        sources.append(source)
        targets.append(target)
        attrs.append([np.sin(dist), np.cos(dist)])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float)
    return Data(edge_index=edge_index, edge_attr=edge_attr)
