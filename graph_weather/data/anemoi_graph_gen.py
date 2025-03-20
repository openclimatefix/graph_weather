import math
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


class AnemoiGrid:
    """
    Implementation of the Anemoi grid structure used by AIFS.

    Creates a reduced Gaussian octahedral grid with attention-based graph connections.
    The grid is designed for scalability and integration into larger systems.
    """

    def __init__(self, grid_resolution: int = 128):
        """
        Initialize an Anemoi grid with the specified resolution.

        Args:
            grid_resolution: Base resolution of the octahedral grid (typically 128, 256, or 512 for weather models)
        """
        self.grid_resolution = grid_resolution

        # Generate the octahedral grid points (simplified Gaussian grid)
        self.vertices, self.lat_bands = self._create_octahedral_grid()

        # Build the graph with attention-based connections
        self.graph = self._build_graph()

    def _create_octahedral_grid(self) -> Tuple[np.ndarray, List[int]]:
        """
        Create a reduced Gaussian octahedral grid.

        Returns:
            A tuple (vertices, latitude_band_sizes):
              - vertices: An array of shape (N, 3) with Cartesian coordinates.
              - latitude_band_sizes: A list indicating the number of points in each latitude band.
        """
        vertices = []
        lat_bands = []

        # Number of latitude bands (half the resolution)
        n_lat = self.grid_resolution // 2

        # Generate points for each latitude band in the northern hemisphere
        for i in range(n_lat):
            # Calculate latitude (Gaussian quadrature points)
            lat = 90 - (i + 0.5) * 180 / n_lat  # degrees

            # Calculate number of points in this latitude band
            n_lon = max(4, int(2 * self.grid_resolution * math.cos(math.radians(lat))))
            n_lon = 4 * math.ceil(n_lon / 4)  # Ensure divisibility by 4 for octahedral structure

            lat_bands.append(n_lon)

            # Generate evenly spaced points along this latitude
            for j in range(n_lon):
                lon = j * 360 / n_lon - 180  # degrees
                x, y, z = self._latlon_to_cartesian(lat, lon)
                vertices.append([x, y, z])

        # Generate points for the southern hemisphere by mirroring the northern hemisphere
        for i in range(n_lat - 1, -1, -1):
            lat = -90 + (i + 0.5) * 180 / n_lat  # degrees
            n_lon = lat_bands[n_lat - i - 1]  # Use the same number of points for symmetry
            lat_bands.append(n_lon)

            for j in range(n_lon):
                lon = j * 360 / n_lon - 180  # degrees
                x, y, z = self._latlon_to_cartesian(lat, lon)
                vertices.append([x, y, z])

        return np.array(vertices), lat_bands

    def _latlon_to_cartesian(self, lat: float, lon: float) -> Tuple[float, float, float]:
        """
        Convert latitude and longitude to 3D Cartesian coordinates on the unit sphere.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.

        Returns:
            Tuple (x, y, z) on the unit sphere.
        """
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x = math.cos(lat_rad) * math.cos(lon_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad)
        z = math.sin(lat_rad)
        return x, y, z

    def _cartesian_to_latlon(self, xyz: np.ndarray) -> Tuple[float, float]:
        """
        Convert 3D Cartesian coordinates to latitude and longitude in degrees.

        Args:
            xyz: A 3D point as a numpy array.

        Returns:
            Tuple (latitude, longitude) in degrees.
        """
        x, y, z = xyz
        lat = math.degrees(math.asin(z))
        lon = math.degrees(math.atan2(y, x))
        return lat, lon

    def _build_graph(self) -> nx.Graph:
        """
        Build the graph structure with attention-based connections.

        The graph includes:
          - Local connections: Nearest neighbors within each latitude band.
          - Inter-band connections: Connections to neighboring latitude bands to mimic the sliding attention window.

        Returns:
            A NetworkX graph representing the grid.
        """
        G = nx.Graph()

        # Add nodes with positional and geographic data
        for i, vertex in enumerate(self.vertices):
            lat, lon = self._cartesian_to_latlon(vertex)
            G.add_node(i, pos=vertex, lat=lat, lon=lon)

        # Add local (intra-band) connections
        start_idx = 0
        for band_size in self.lat_bands:
            for i in range(band_size):
                # Connect each node to immediate neighbors (with wrap-around)
                G.add_edge(start_idx + i, start_idx + (i + 1) % band_size)
                G.add_edge(start_idx + i, start_idx + (i - 1) % band_size)

                # Also connect to second neighbors for a wider receptive field
                G.add_edge(start_idx + i, start_idx + (i + 2) % band_size)
                G.add_edge(start_idx + i, start_idx + (i - 2) % band_size)
            start_idx += band_size

        # Add inter-band connections (simulate sliding attention window)
        start_idx_prev = 0
        for band_idx, band_size in enumerate(self.lat_bands[:-1]):
            start_idx_curr = start_idx_prev + band_size
            band_size_next = self.lat_bands[band_idx + 1]

            # Connect each point in the current band to nearest points in the adjacent band
            for i in range(band_size):
                ratio = band_size_next / band_size
                j1 = int(i * ratio) % band_size_next
                j2 = int((i + 0.5) * ratio) % band_size_next
                G.add_edge(start_idx_prev + i, start_idx_curr + j1)
                G.add_edge(start_idx_prev + i, start_idx_curr + j2)
            start_idx_prev = start_idx_curr

        return G

    def get_graph(self) -> nx.Graph:
        """
        Return the NetworkX graph representation of the grid.

        Returns:
            A NetworkX graph.
        """
        return self.graph

    def get_vertices(self) -> np.ndarray:
        """
        Return the vertex coordinates of the grid.

        Returns:
            A numpy array of shape (N, 3) representing Cartesian coordinates.
        """
        return self.vertices

    def get_latlon_coordinates(self) -> List[Tuple[float, float]]:
        """
        Return all vertex coordinates as (latitude, longitude) pairs.

        Returns:
            A list of tuples containing latitude and longitude.
        """
        return [self._cartesian_to_latlon(v) for v in self.vertices]

    def get_adjacency_info(self) -> Tuple[List[int], List[int]]:
        """
        Get adjacency information for the graph in edge list format.

        Returns:
            A tuple (src_nodes, dst_nodes) representing the source and destination nodes of each edge.
        """
        src_nodes = []
        dst_nodes = []
        for u, v in self.graph.edges():
            src_nodes.append(u)
            dst_nodes.append(v)
            # Include reverse edge for undirected graph
            src_nodes.append(v)
            dst_nodes.append(u)
        return src_nodes, dst_nodes

    def get_sliding_window_attention_indices(self, window_size: int = 40) -> List[List[int]]:
        """
        Generate sliding window attention indices for AIFS-style processing.

        Args:
            window_size: Size of the attention window (typically 40).

        Returns:
            A list of lists, where each sublist contains node indices for one attention window.
        """
        attention_windows = []
        n_vertices = len(self.vertices)

        # Create overlapping windows to process the grid as a sequence
        for i in range(0, n_vertices, window_size // 2):
            window = list(range(i, min(i + window_size, n_vertices)))
            if len(window) >= window_size // 2:
                attention_windows.append(window)

        return attention_windows


class AnemoiGraphAdapter:
    """
    Adapter class to integrate Anemoi graphs with machine learning frameworks.

    This class adapts the Anemoi grid for use in AIFS-like architectures, providing methods
    to map input data to graph node features, prepare model inputs, and process model outputs.
    """

    def __init__(self, resolution: int = 128):
        """
        Initialize the Anemoi graph adapter with a specified grid resolution.

        Args:
            resolution: The grid resolution used for generating the Anemoi grid.
        """
        self.grid = AnemoiGrid(grid_resolution=resolution)
        self.graph = self.grid.get_graph()

        # Cache common properties for efficiency
        self.n_nodes = len(self.graph.nodes())
        self.latlon = self.grid.get_latlon_coordinates()
        self.attention_windows = self.grid.get_sliding_window_attention_indices()

    def get_node_features(self, data: np.ndarray, variable_idx: int = 0) -> np.ndarray:
        """
        Map input data to graph node features.

        This implementation uses a nearest-neighbor lookup to assign values from the input data array
        (with dimensions [variables, lat, lon]) to nodes based on their geographic locations.

        Args:
            data: Input data array with shape [variables, lat, lon].
            variable_idx: Index of the variable to map (default: 0).

        Returns:
            A numpy array of shape [n_nodes] containing node feature values.
        """
        features = np.zeros(self.n_nodes)

        for i, (lat, lon) in enumerate(self.latlon):
            lat_idx = int((90 - lat) / 180 * data.shape[1])
            lon_idx = int((lon + 180) / 360 * data.shape[2])
            lat_idx = max(0, min(lat_idx, data.shape[1] - 1))
            lon_idx = max(0, min(lon_idx, data.shape[2] - 1))
            features[i] = data[variable_idx, lat_idx, lon_idx]

        return features

    def prepare_model_inputs(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for an AIFS-like model.

        Maps multiple input variables to node features, constructs the graph connectivity, and
        generates attention masks based on sliding windows.

        Args:
            data: Dictionary mapping variable names to data arrays.

        Returns:
            Dictionary containing:
              - 'node_features': Array of shape [n_nodes, n_variables].
              - 'edge_index': 2D array representing graph connectivity.
              - 'attention_masks': Array of attention masks for sliding windows.
              - 'positions': Cartesian coordinates of grid vertices.
        """
        node_features = {}
        for i, (var_name, var_data) in enumerate(data.items()):
            node_features[var_name] = self.get_node_features(var_data, i)

        features = np.stack([node_features[var] for var in data.keys()], axis=1)
        src_nodes, dst_nodes = self.grid.get_adjacency_info()

        attention_masks = np.zeros((len(self.attention_windows), self.n_nodes), dtype=np.float32)
        for i, window in enumerate(self.attention_windows):
            attention_masks[i, window] = 1.0

        return {
            "node_features": features,
            "edge_index": np.array([src_nodes, dst_nodes]),
            "attention_masks": attention_masks,
            "positions": self.grid.get_vertices(),
        }

    def process_model_outputs(self, outputs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process model outputs and regrid them back to a regular 2D grid.

        This simplified implementation performs nearest-neighbor regridding from node outputs to a 2D grid.

        Args:
            outputs: Model output array with shape [n_nodes, n_variables].

        Returns:
            Dictionary mapping variable names to 2D arrays representing regridded outputs.
        """
        n_vars = outputs.shape[1]
        n_lat, n_lon = 180, 360  # Example grid dimensions

        regridded = {}
        for var_idx in range(n_vars):
            grid = np.zeros((n_lat, n_lon))
            for i, (lat, lon) in enumerate(self.latlon):
                lat_idx = int((90 - lat) / 180 * n_lat)
                lon_idx = int((lon + 180) / 360 * n_lon)
                lat_idx = max(0, min(lat_idx, n_lat - 1))
                lon_idx = max(0, min(lon_idx, n_lon - 1))
                grid[lat_idx, lon_idx] = outputs[i, var_idx]
            regridded[f"var_{var_idx}"] = grid

        return regridded
