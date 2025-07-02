"""Build the three graphs for GenCast.

The following code is a port of several components from GraphCast's original graph generation
(https://github.com/google-deepmind/graphcast) to PyG and PyTorch. The graphs are:
- g2m: grid to mesh.
- mesh: icosphere refinement.
- m2g: mesh to grid.
- khop: k-hop neighbours mesh.
"""

import gc

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData

# from torch_geometric.transforms import TwoHop
from graph_weather.models.gencast.graph import grid_mesh_connectivity, icosahedral_mesh, model_utils

# Some configs from graphcast:
_spatial_features_kwargs = dict(
    add_node_positions=False,
    add_node_latitude=True,
    add_node_longitude=True,
    add_relative_positions=True,
    relative_longitude_local_coordinates=True,
    relative_latitude_local_coordinates=True,
)

# radius_query_fraction_edge_length: Scalar that will be multiplied by the
#   length of the longest edge of the finest mesh to define the radius of
#   connectivity to use in the Grid2Mesh graph. Reasonable values are
#   between 0.6 and 1. 0.6 reduces the number of grid points feeding into
#   multiple mesh nodes and therefore reduces edge count and memory use, but
#   1 gives better predictions.
# mesh2grid_edge_normalization_factor: Allows explicitly controlling edge
#   normalization for mesh2grid edges. If None, defaults to max edge length.
#   This supports using pre-trained model weights with a different graph
#   structure to what it was trained on.

radius_query_fraction_edge_length = 0.6
mesh2grid_edge_normalization_factor = None


def _get_max_edge_distance(mesh):
    senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
    edge_distances = np.linalg.norm(mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
    return edge_distances.max()


class GraphBuilder:
    """
    Class for building GenCast's graphs.

    Attributes:
        g2m_graph (pyg.data.HeteroData): heterogeneous directed graph connecting the grid nodes
            to the mesh nodes.
        mesh_graph (pyg.data.Data): undirected graph connecting the mesh nodes.
        m2g_graph (pyg.data.HeteroData): heterogeneous directed graph connecting the mesh nodes
            to the grid nodes.
        khop_mesh_graph (pyg.data.Data): augmented version of mesh_graph in which every node is
            connected to its num_hops neighbours.
        grid_nodes_dim (int): dimension of the grid nodes features.
        mesh_nodes_dim (int): dimension of the mesh nodes features.
        mesh_edges_dim (int): dimension of the mesh edges features.
        g2m_edges_dim (int): dimension of the "grid to mesh" edges features.
        m2g_edges_dim (int): dimension of the "mesh to grid" edges features.
    """

    def __init__(
        self,
        grid_lon: np.ndarray,
        grid_lat: np.ndarray,
        splits: int = 5,
        num_hops: int = 0,
        device: torch.device = torch.device("cpu"),
        khop_device: torch.device = torch.device("cpu"),
        add_edge_features_to_khop=True,
    ):
        """Initialize the GraphBuilder object.

        Args:
            grid_lon (np.ndarray): 1D np.ndarray containing the list of longitudes.
            grid_lat: (np.ndarray) 1D np.ndarray containing the list of latitudes.
            splits (int): number of times to split the icosphere to build the mesh. Defaults to 5.
            num_hops (int): if num_hops=k then khop_mesh_graph will be the k-neighbours version of
                the mesh. Defaults to 0.
            device: the device to which the final graph will be moved.
            khop_device: the device that will compute the k-hop mesh graph. Note that while setting
                this to gpu may result in faster computations, it may also cause some memory leaks
                in the current implementation. Defaults to cpu.
            add_edge_features_to_khop (bool): if true compute edge features for the k-hop neighbours
                graph. Defaults to False.
        """

        self._spatial_features_kwargs = _spatial_features_kwargs
        self.add_edge_features_to_khop = add_edge_features_to_khop
        self.device = device
        self.khop_device = khop_device

        # Specification of the mesh.
        _icosahedral_refinements = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
            splits
        )
        self._mesh = _icosahedral_refinements[-1]

        # Obtain the query radius in absolute units for the unit-sphere for the
        # grid2mesh model, by rescaling the `radius_query_fraction_edge_length`.
        self._query_radius = _get_max_edge_distance(self._mesh) * radius_query_fraction_edge_length
        self._mesh2grid_edge_normalization_factor = mesh2grid_edge_normalization_factor

        self.grid_nodes_dim = None
        self.mesh_nodes_dim = None
        self.mesh_edges_dim = None
        self.g2m_edges_dim = None
        self.m2g_edges_dim = None

        # A "_init_mesh_properties":
        # This one could be initialized at init but we delay it for consistency too.
        self._num_mesh_nodes = None  # num_mesh_nodes
        self._mesh_nodes_lat = None  # [num_mesh_nodes]
        self._mesh_nodes_lon = None  # [num_mesh_nodes]

        # A "_init_grid_properties":
        self._grid_lat = None  # [num_lat_points]
        self._grid_lon = None  # [num_lon_points]
        self._num_grid_nodes = None  # num_lat_points * num_lon_points
        self._grid_nodes_lat = None  # [num_grid_nodes]
        self._grid_nodes_lon = None  # [num_grid_nodes]

        self._init_grid_properties(grid_lat, grid_lon)
        self._init_mesh_properties()
        self.g2m_graph = self._init_grid2mesh_graph()
        self.mesh_graph = self._init_mesh_graph()
        self.m2g_graph = self._init_mesh2grid_graph()

        self.num_hops = num_hops
        self.khop_mesh_graph = self._init_khop_mesh_graph()

    def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
        """Inits static properties that have to do with grid nodes."""
        self._grid_lat = grid_lat.astype(np.float32)
        self._grid_lon = grid_lon.astype(np.float32)
        # Initialized the counters.
        self._num_grid_nodes = grid_lat.shape[0] * grid_lon.shape[0]

        # Initialize lat and lon for the grid.
        grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_lon, grid_lat)
        self._grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
        self._grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

    def _init_mesh_properties(self):
        """Inits static properties that have to do with mesh nodes."""
        self._num_mesh_nodes = self._mesh.vertices.shape[0]
        mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
            self._mesh.vertices[:, 0], self._mesh.vertices[:, 1], self._mesh.vertices[:, 2]
        )
        (
            mesh_nodes_lat,
            mesh_nodes_lon,
        ) = model_utils.spherical_to_lat_lon(phi=mesh_phi, theta=mesh_theta)
        # Convert to f32 to ensure the lat/lon features aren't in f64.
        self._mesh_nodes_lat = mesh_nodes_lat.astype(np.float32)
        self._mesh_nodes_lon = mesh_nodes_lon.astype(np.float32)

    def _init_grid2mesh_graph(self):
        """Build Grid2Mesh graph."""

        # Create some edges according to distance between mesh and grid nodes.
        assert self._grid_lat is not None and self._grid_lon is not None
        (grid_indices, mesh_indices) = grid_mesh_connectivity.radius_query_indices(
            grid_latitude=self._grid_lat,
            grid_longitude=self._grid_lon,
            mesh=self._mesh,
            radius=self._query_radius,
        )

        # Edges sending info from grid to mesh.
        senders = grid_indices
        receivers = mesh_indices

        # Precompute structural node and edge features according to config options.
        # Structural features are those that depend on the fixed values of the
        # latitude and longitudes of the nodes.
        (senders_node_features, receivers_node_features, edge_features) = (
            model_utils.get_bipartite_graph_spatial_features(
                senders_node_lat=self._grid_nodes_lat,
                senders_node_lon=self._grid_nodes_lon,
                receivers_node_lat=self._mesh_nodes_lat,
                receivers_node_lon=self._mesh_nodes_lon,
                senders=senders,
                receivers=receivers,
                edge_normalization_factor=None,
                **self._spatial_features_kwargs,
            )
        )

        self.grid_nodes_dim = senders_node_features.shape[1]
        self.mesh_nodes_dim = receivers_node_features.shape[1]
        self.g2m_edges_dim = edge_features.shape[1]

        g2m_graph = HeteroData()
        g2m_graph["grid_nodes"].x = torch.tensor(
            senders_node_features, dtype=torch.float32, device=self.device
        )  # TODO: generate graph with torch or np?
        g2m_graph["mesh_nodes"].x = torch.tensor(
            receivers_node_features, dtype=torch.float32, device=self.device
        )
        g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_index = torch.tensor(
            np.stack([senders, receivers]), dtype=torch.long, device=self.device
        )
        g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_attr = torch.tensor(
            edge_features, dtype=torch.float32, device=self.device
        )

        return g2m_graph

    def _init_mesh_graph(self):
        """Build Mesh graph."""
        # Work simply on the mesh edges.
        senders, receivers = icosahedral_mesh.faces_to_edges(self._mesh.faces)

        # Precompute structural node and edge features according to config options.
        # Structural features are those that depend on the fixed values of the
        # latitude and longitudes of the nodes.
        assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
        node_features, edge_features = model_utils.get_graph_spatial_features(
            node_lat=self._mesh_nodes_lat,
            node_lon=self._mesh_nodes_lon,
            senders=senders,
            receivers=receivers,
            **self._spatial_features_kwargs,
        )

        self.mesh_edges_dim = edge_features.shape[1]

        mesh_graph = Data(
            x=torch.tensor(node_features, dtype=torch.float32, device=self.device),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32, device=self.device),
            edge_index=torch.tensor(
                np.stack([senders, receivers]), dtype=torch.long, device=self.device
            ),
        )

        return mesh_graph

    def _init_mesh2grid_graph(self):
        """Build Mesh2Grid graph."""

        # Create some edges according to how the grid nodes are contained by
        # mesh triangles.
        (grid_indices, mesh_indices) = grid_mesh_connectivity.in_mesh_triangle_indices(
            grid_latitude=self._grid_lat, grid_longitude=self._grid_lon, mesh=self._mesh
        )

        # Edges sending info from mesh to grid.
        senders = mesh_indices
        receivers = grid_indices

        # Precompute structural node and edge features according to config options.
        assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
        (senders_node_features, receivers_node_features, edge_features) = (
            model_utils.get_bipartite_graph_spatial_features(
                senders_node_lat=self._mesh_nodes_lat,
                senders_node_lon=self._mesh_nodes_lon,
                receivers_node_lat=self._grid_nodes_lat,
                receivers_node_lon=self._grid_nodes_lon,
                senders=senders,
                receivers=receivers,
                edge_normalization_factor=self._mesh2grid_edge_normalization_factor,
                **self._spatial_features_kwargs,
            )
        )

        self.m2g_edges_dim = edge_features.shape[1]

        m2g_graph = HeteroData()
        m2g_graph["mesh_nodes"].x = torch.tensor(
            senders_node_features, dtype=torch.float32, device=self.device
        )
        m2g_graph["grid_nodes"].x = torch.tensor(
            receivers_node_features, dtype=torch.float32, device=self.device
        )
        m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_index = torch.tensor(
            np.stack([senders, receivers]), dtype=torch.long, device=self.device
        )
        m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_attr = torch.tensor(
            edge_features, dtype=torch.float32, device=self.device
        )

        return m2g_graph

    def _init_khop_mesh_graph(self):
        """Build k-hop Mesh graph.

        This implementation constructs the sparse adjacency matrix associated with the mesh graph
        and computes its powers in a sparse manner.
        """

        # PyG version:
        # transform = TwoHop()
        # khop_mesh_graph = self.mesh_graph
        # for _ in range(self.num_hops):
        #    khop_mesh_graph = transform(khop_mesh_graph)

        # build the sparse adjacency matrix
        edge_index = self.mesh_graph.edge_index
        adj = torch.sparse_coo_tensor(
            edge_index,
            values=torch.ones_like(edge_index[0], dtype=torch.float32),
            size=(self._num_mesh_nodes, self._num_mesh_nodes),
        ).to(
            self.khop_device
        )  # cpu is more memory-efficient, why?

        adj_k = adj.coalesce()
        for _ in range(self.num_hops - 1):
            # add to previous edges their 1-hop neighbours
            adj_k = adj_k + torch.sparse.mm(adj_k, adj)
            adj_k = adj_k.coalesce()

            # remove self loops
            new_indices = adj_k.indices()
            mask = new_indices[0] != new_indices[1]
            new_indices = new_indices[:, mask]
            new_values = torch.ones_like(adj_k.values()[mask])

            # build k-hop sparse matrix
            adj_k = torch.sparse_coo_tensor(
                indices=new_indices, values=new_values, size=adj_k.shape
            ).coalesce()

            # clean memory
            del mask, new_indices, new_values
            gc.collect()

        # build k-hop graph
        khop_mesh_graph = Data(x=self.mesh_graph.x, edge_index=adj_k.indices().to(self.device))
        del adj_k, adj
        gc.collect()

        # optionally compute edges' features: computationally expensive for a big mesh!
        if self.add_edge_features_to_khop:
            senders = khop_mesh_graph.edge_index[0].cpu()
            receivers = khop_mesh_graph.edge_index[1].cpu()
            _, edge_features = model_utils.get_graph_spatial_features(
                node_lat=self._mesh_nodes_lat,
                node_lon=self._mesh_nodes_lon,
                senders=senders,
                receivers=receivers,
                **self._spatial_features_kwargs,
            )
            khop_mesh_graph.edge_attr = torch.tensor(
                edge_features, dtype=torch.float32, device=self.device
            )
        return khop_mesh_graph
