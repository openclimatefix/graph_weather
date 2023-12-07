"""
Creating geodesic icosahedron with given (integer) subdivision frequency (and
not by recursively applying Loop-like subdivision).

Advantage of subdivision frequency compared to the recursive subdivision is in
controlling the mesh resolution. Mesh resolution grows quadratically with
subdivision frequencies while it grows exponentially with iterations of the
recursive subdivision. To be precise, using the recursive
subdivision (each iteration being a subdivision with frequency nu=2), the
possible number of vertices grows with iterations i as
    [12+10*(2**i+1)*(2**i-1) for i in range(10)]
which gives
    [12, 42, 162, 642, 2562, 10242, 40962, 163842, 655362, 2621442].
Notice for example there is no mesh having between 2562 and 10242 vertices.
Using subdivision frequency, possible number of vertices grows with nu as
    [12+10*(nu+1)*(nu-1) for nu in range(1,33)]
which gives
    [12, 42, 92, 162, 252, 362, 492, 642, 812, 1002, 1212, 1442, 1692, 1962,
     2252, 2562, 2892, 3242, 3612, 4002, 4412, 4842, 5292, 5762, 6252, 6762,
     7292, 7842, 8412, 9002, 9612, 10242]
where nu = 32 gives 10242 vertices, and there are 15 meshes having between
2562 and 10242 vertices. The advantage is even more pronounced when using
higher resolutions.

Author: vand@dtu.dk, 2014, 2017, 2021.
Originally developed in connectiton with
https://ieeexplore.ieee.org/document/7182720

This code is copied in as there is an improvement in the inside_points function that
is not merged in that speeds up generation 5-8x. See https://github.com/vedranaa/icosphere/pull/3

"""

import numpy as np
import torch
from torch_geometric.data import Data


def icosphere(nu=1, nr_verts=None):
    """
    Returns a geodesic icosahedron with subdivision frequency nu. Frequency
    nu = 1 returns regular unit icosahedron, and nu>1 preformes subdivision.
    If nr_verts is given, nu will be adjusted such that icosphere contains
    at least nr_verts vertices. Returned faces are zero-indexed!

    Parameters
    ----------
    nu : subdivision frequency, integer (larger than 1 to make a change).
    nr_verts: desired number of mesh vertices, if given, nu may be increased.


    Returns
    -------
    subvertices : vertex list, numpy array of shape (20+10*(nu+1)*(nu-1)/2, 3)
    subfaces : face list, numpy array of shape (10*n**2, 3)

    """

    # Unit icosahedron
    (vertices, faces) = icosahedron()

    # If nr_verts given, computing appropriate subdivision frequency nu.
    # We know nr_verts = 12+10*(nu+1)(nu-1)
    if not nr_verts is None:
        nu_min = np.ceil(np.sqrt(max(1 + (nr_verts - 12) / 10, 1)))
        nu = max(nu, nu_min)

    # Subdividing
    if nu > 1:
        (vertices, faces) = subdivide_mesh(vertices, faces, nu)
        vertices = vertices / np.sqrt(np.sum(vertices**2, axis=1, keepdims=True))

    return (vertices, faces)


def icosahedron():
    """' Regular unit icosahedron."""

    # 12 principal directions in 3D space: points on an unit icosahedron
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array(
        [[0, 1, phi], [0, -1, phi], [1, phi, 0], [-1, phi, 0], [phi, 0, 1], [-phi, 0, 1]]
    ) / np.sqrt(1 + phi**2)
    vertices = np.r_[vertices, -vertices]

    # 20 faces
    faces = np.array(
        [
            [0, 5, 1],
            [0, 3, 5],
            [0, 2, 3],
            [0, 4, 2],
            [0, 1, 4],
            [1, 5, 8],
            [5, 3, 10],
            [3, 2, 7],
            [2, 4, 11],
            [4, 1, 9],
            [7, 11, 6],
            [11, 9, 6],
            [9, 8, 6],
            [8, 10, 6],
            [10, 7, 6],
            [2, 11, 7],
            [4, 9, 11],
            [1, 8, 9],
            [5, 10, 8],
            [3, 7, 10],
        ],
        dtype=int,
    )

    return (vertices, faces)


def subdivide_mesh(vertices, faces, nu):
    """
    Subdivides mesh by adding vertices on mesh edges and faces. Each edge
    will be divided in nu segments. (For example, for nu=2 one vertex is added
    on each mesh edge, for nu=3 two vertices are added on each mesh edge and
    one vertex is added on each face.) If V and F are number of mesh vertices
    and number of mesh faces for the input mesh, the subdivided mesh contains
    V + F*(nu+1)*(nu-1)/2 vertices and F*nu^2 faces.

    Parameters
    ----------
    vertices : vertex list, numpy array of shape (V,3)
    faces : face list, numby array of shape (F,3). Zero indexed.
    nu : subdivision frequency, integer (larger than 1 to make a change).

    Returns
    -------
    subvertices : vertex list, numpy array of shape (V + F*(nu+1)*(nu-1)/2, 3)
    subfaces : face list, numpy array of shape (F*n**2, 3)

    Author: vand at dtu.dk, 8.12.2017. Translated to python 6.4.2021

    """

    edges = np.r_[faces[:, :-1], faces[:, 1:], faces[:, [0, 2]]]
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    F = faces.shape[0]
    V = vertices.shape[0]
    E = edges.shape[0]
    subfaces = np.empty((F * nu**2, 3), dtype=int)
    subvertices = np.empty((V + E * (nu - 1) + F * (nu - 1) * (nu - 2) // 2, 3))

    subvertices[:V] = vertices

    # Dictionary for accessing edge index from indices of edge vertices.
    edge_indices = dict()
    for i in range(V):
        edge_indices[i] = dict()
    for i in range(E):
        edge_indices[edges[i, 0]][edges[i, 1]] = i
        edge_indices[edges[i, 1]][edges[i, 0]] = -i

    template = faces_template(nu)
    ordering = vertex_ordering(nu)
    reordered_template = ordering[template]

    # At this point, we have V vertices, and now we add (nu-1) vertex per edge
    # (on-edge vertices).
    w = np.arange(1, nu) / nu  # interpolation weights
    for e in range(E):
        edge = edges[e]
        for k in range(nu - 1):
            subvertices[V + e * (nu - 1) + k] = (
                w[-1 - k] * vertices[edge[0]] + w[k] * vertices[edge[1]]
            )

    # At this point we have E(nu-1)+V vertices, and we add (nu-1)*(nu-2)/2
    # vertices per face (on-face vertices).
    r = np.arange(nu - 1)
    for f in range(F):
        # First, fixing connectivity. We get hold of the indices of all
        # vertices invoved in this subface: original, on-edges and on-faces.
        T = np.arange(
            f * (nu - 1) * (nu - 2) // 2 + E * (nu - 1) + V,
            (f + 1) * (nu - 1) * (nu - 2) // 2 + E * (nu - 1) + V,
        )  # will be added
        eAB = edge_indices[faces[f, 0]][faces[f, 1]]
        eAC = edge_indices[faces[f, 0]][faces[f, 2]]
        eBC = edge_indices[faces[f, 1]][faces[f, 2]]
        AB = reverse(abs(eAB) * (nu - 1) + V + r, eAB < 0)  # already added
        AC = reverse(abs(eAC) * (nu - 1) + V + r, eAC < 0)  # already added
        BC = reverse(abs(eBC) * (nu - 1) + V + r, eBC < 0)  # already added
        VEF = np.r_[faces[f], AB, AC, BC, T]
        subfaces[f * nu**2 : (f + 1) * nu**2, :] = VEF[reordered_template]
        # Now geometry, computing positions of face vertices.
        subvertices[T, :] = inside_points(subvertices[AB, :], subvertices[AC, :])

    return (subvertices, subfaces)


def reverse(vector, flag):
    """' For reversing the direction of an edge."""

    if flag:
        vector = vector[::-1]
    return vector


def faces_template(nu):
    """
    Template for linking subfaces                  0
    in a subdivision of a face.                   / \
    Returns faces with vertex                    1---2
    indexing given by reading order             / \ / \
    (as illustratated).                        3---4---5
                                              / \ / \ / \
                                             6---7---8---9
                                            / \ / \ / \ / \
                                           10--11--12--13--14
    """

    faces = []
    # looping in layers of triangles
    for i in range(nu):
        vertex0 = i * (i + 1) // 2
        skip = i + 1
        for j in range(i):  # adding pairs of triangles, will not run for i==0
            faces.append([j + vertex0, j + vertex0 + skip, j + vertex0 + skip + 1])
            faces.append([j + vertex0, j + vertex0 + skip + 1, j + vertex0 + 1])
        # adding the last (unpaired, rightmost) triangle
        faces.append([i + vertex0, i + vertex0 + skip, i + vertex0 + skip + 1])

    return np.array(faces)


def vertex_ordering(nu):
    """
    Permutation for ordering of                    0
    face vertices which transformes               / \
    reading-order indexing into indexing         3---6
    first corners vertices, then on-edges       / \ / \
    vertices, and then on-face vertices        4---12--7
    (as illustrated).                         / \ / \ / \
                                             5---13--14--8
                                            / \ / \ / \ / \
                                           1---9--10--11---2
    """

    left = [j for j in range(3, nu + 2)]
    right = [j for j in range(nu + 2, 2 * nu + 1)]
    bottom = [j for j in range(2 * nu + 1, 3 * nu)]
    inside = [j for j in range(3 * nu, (nu + 1) * (nu + 2) // 2)]

    o = [0]  # topmost corner
    for i in range(nu - 1):
        o.append(left[i])
        o = o + inside[i * (i - 1) // 2 : i * (i + 1) // 2]
        o.append(right[i])
    o = o + [1] + bottom + [2]

    return np.array(o)


def inside_points(vAB, vAC):
    """
    Returns coordinates of the inside                 .
    (on-face) vertices (marked by star)              / \
    for subdivision of the face ABC when         vAB0---vAC0
    given coordinates of the on-edge               / \ / \
    vertices  AB[i] and AC[i].                 vAB1---*---vAC1
                                                 / \ / \ / \
                                             vAB2---*---*---vAC2
                                               / \ / \ / \ / \
                                              .---.---.---.---.
    """
    out = []
    u = vAB.shape[0]
    for i in range(0 if u == 1 else 1, u):
        # Linearly interpolate between vABi and vACi in `i + 1` (`j`) steps,
        # not including the endpoints.
        # This could be written as
        #   vABi = vAB[i, :]
        #   vACi = vAC[i, :]
        #   interp_multipliers = np.arange(1, j) / j
        #   res = np.outer(interp_multipliers, vACi) + np.outer(1 - interp_multipliers, vABi)
        # but that will involve some extra work on `np.outer`'s part that we can
        # do ourselves since we know the shapes we're working with.
        j = i + 1
        interp_multipliers = (np.arange(1, j) / j)[:, None]
        out.append(
            np.multiply(interp_multipliers, vAC[i, None]) +
            np.multiply(1 - interp_multipliers, vAB[i, None])
        )
    return np.concatenate(out)


def generate_icosphere_graph(resolution=1):
    """
    Generate a graph of the icosphere with the given level of subdivision.
    """
    vertices, faces = icosphere(resolution)
    edges = np.r_[faces[:, :-1], faces[:, 1:], faces[:, [0, 2]]]
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return vertices, edges

def generate_icosphere_mapping(lat_lons, resolutions=(1, 2, 4, 8, 16, 32, 64)) -> Data:
    """
    Generate mapping from lat/lon to icosphere index.

    GraphCast maps from lat/lon directly to all the different icosphere levels
    And maps back from all te different icosphere levels to lat/lon
    Nodes are shared to an extant, in that the base nodes (12) are connected to all the other layers and has more
    incoming edges than each of the further refined ones. In the most extreme case, the finest resolution
    icosphere nodes have only the incoming edges from the nearby nodes. For 6 different layers, each layer adds 5
    more incoming edges to the nodes in the above layers. So the top layer nodes have 30 incoming edges, for the finest
    one, it has 5 incoming edges. This is a bipartite graph, so the edges are between the lat/lon nodes and the icosphere

    Args:
        lat_lons: List of (lat,lon) points
        resolutions: Icosphere resolution levels, first 7 levels correspond to Graphcast levels, in ascending order of resolution
    """
    num_latlons = len(lat_lons)
    verticies_per_level = []
    edges_per_level = []
    for resolution in resolutions:
        vertices, edges = generate_icosphere_graph(resolution)
        verticies_per_level.append(vertices)
        edges_per_level.append(edges)

    # Check the verticies of each pair are the same up to the resolution
    for i in range(len(verticies_per_level) - 1):
        for vertex_lower_index, vertex in enumerate(verticies_per_level[i]):
            vertex_mapping = np.argmin(np.sum(np.abs(verticies_per_level[i + 1] - vertex), axis=1), axis=0)
            # Change all edge indicies from vertex_lower_index to vertex_mapping
            edges_per_level[i + 1][edges_per_level[i + 1] == vertex_lower_index] = vertex_mapping
    verticies = verticies_per_level[-1] # The last layer has all the verticies of the ones above
    edges = np.unique(np.sort(np.concatenate(edges_per_level), axis=1), axis=0)
    # TODO Create mapping from the lat/lon to the icosphere nodes
    ico_graph = Data(pos=torch.tensor(verticies, dtype=torch.float),
                     edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())
    return ico_graph


def generate_latent_ico_graph(h3_mapping, h3_distances):
    """
    Generate latent h3 graph.

    Args:
        base_h3_map: Mapping from h3 index to index in latent graph
        h3_mapping: Mapping from lat/lon to h3 index
        h3_distances: Distances between lat/lon and h3 index

    Returns:
        Latent h3 graph
    """
    # Get connectivity of the graph
    edge_sources = []
    edge_targets = []
    edge_attrs = []
    for h3_index in h3_mapping:
        h_points = h3.k_ring(h3_index, 1)
        for h in h_points:  # Already includes itself
            distance = h3.point_dist(h3.h3_to_geo(h3_index), h3.h3_to_geo(h), unit="rads")
            edge_attrs.append([np.sin(distance), np.cos(distance)])
            edge_sources.append(h3_mapping[h3_index])
            edge_targets.append(h3_mapping[h])
    edge_sources = np.array(edge_sources)
    edge_targets = np.array(edge_targets)
    edge_attrs = np.array(edge_attrs)
    return edge_sources, edge_targets, edge_attrs
