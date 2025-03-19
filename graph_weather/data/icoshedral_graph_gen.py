import numpy as np
import networkx as nx
from typing import List, Tuple, Dict

class IcosahedralGrid:
    """
    Generate an icosahedral grid on a sphere for weather/climate modeling.

    This class implements a canonical icosahedron and supports optional subdivisions 
    to refine the grid. It computes the vertices, faces, and constructs a corresponding 
    NetworkX graph representation of the grid. The vertices are projected onto the unit 
    sphere, and geographic coordinates (latitude, longitude) are derived from the Cartesian 
    coordinates.

    Attributes:
        subdivision_level (int): Number of times the base icosahedron is subdivided to increase resolution.
        vertices (np.ndarray): An array of shape (N, 3) containing the 3D coordinates of each vertex.
        faces (List[Tuple[int, int, int]]): A list of tuples defining triangular faces using vertex indices.
        graph (nx.Graph): A NetworkX graph representation of the grid.
    """

    def __init__(self, subdivision_level: int = 0):
        """
        Initialize an icosahedral grid with a specified subdivision level.

        The grid is constructed by generating the base icosahedron (with canonical vertex ordering 
        and face definitions), optionally subdividing its faces to refine the grid, projecting all 
        vertices onto a unit sphere, and finally building a graph representation.

        Args:
            subdivision_level (int): The number of subdivision iterations to apply. Each subdivision 
                                     splits each triangular face into 4 smaller triangles.
        """
        self.subdivision_level = subdivision_level
        
        # Generate the canonical base icosahedron vertices and faces
        self.vertices = self._create_base_icosahedron()
        self.faces = self._create_base_faces()
        
        # Subdivide the faces if a higher resolution grid is required
        for _ in range(self.subdivision_level):
            self._subdivide()
            
        # Ensure all vertices lie exactly on the unit sphere
        self.vertices = self._project_to_sphere(self.vertices)
        
        # Build a graph representation where nodes are vertices and edges are derived from faces
        self.graph = self._build_graph()
        
    def _create_base_icosahedron(self) -> np.ndarray:
        """
        Create the 12 vertices of a regular icosahedron using canonical vertex ordering.

        The vertices are defined using the golden ratio (phi) and then projected onto the unit sphere.
        This canonical ordering is chosen to be consistent with standard icosahedron definitions.

        Returns:
            np.ndarray: A (12, 3) array of vertex coordinates.
        """
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = np.array([
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],
            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],
            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ], dtype=np.float64)
        
        return self._project_to_sphere(vertices)
    
    def _create_base_faces(self) -> List[Tuple[int, int, int]]:
        """
        Create the 20 triangular faces of the icosahedron using canonical indices.

        Each face is defined as a tuple of three vertex indices. The provided face definitions 
        are standard for a canonical icosahedron, ensuring that the grid will have 20 faces and 
        30 unique edges.

        Returns:
            List[Tuple[int, int, int]]: A list containing 20 tuples, each with three integer indices.
        """
        return [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (8, 7, 1)
        ]
    
    def _project_to_sphere(self, vertices: np.ndarray) -> np.ndarray:
        """
        Project a set of vertices onto the unit sphere.

        This function normalizes each vertex (treating it as a 3D vector) so that its length becomes 1.

        Args:
            vertices (np.ndarray): An array of vertex coordinates.

        Returns:
            np.ndarray: An array of normalized vertex coordinates.
        """
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        normalized = vertices / norms
        return normalized
    
    def _subdivide(self):
        """
        Subdivide each triangular face by adding vertices at edge midpoints.

        This method refines the grid by iterating over each face, computing midpoints for each edge 
        (with caching to avoid duplicates), and forming four new faces from the original triangle.
        """
        edge_midpoints = {}  # Cache to store computed midpoints
        new_faces = []
        
        def get_midpoint(i, j):
            """
            Get the index of the midpoint for the edge between vertices i and j.

            If the midpoint does not exist, compute it, add it to the vertices array, and cache the index.

            Args:
                i (int): Index of the first vertex.
                j (int): Index of the second vertex.

            Returns:
                int: Index of the midpoint vertex.
            """
            key = tuple(sorted((i, j)))
            if key in edge_midpoints:
                return edge_midpoints[key]
            else:
                vi = self.vertices[i]
                vj = self.vertices[j]
                midpoint = (vi + vj) / 2.0
                midpoint = midpoint / np.linalg.norm(midpoint)  # Ensure the midpoint is on the unit sphere
                self.vertices = np.vstack([self.vertices, midpoint])
                idx = len(self.vertices) - 1
                edge_midpoints[key] = idx
                return idx
        
        for face in self.faces:
            i, j, k = face
            a = get_midpoint(i, j)
            b = get_midpoint(j, k)
            c = get_midpoint(k, i)
            
            # Create four new faces from the original triangle
            new_faces.append((i, a, c))
            new_faces.append((j, b, a))
            new_faces.append((k, c, b))
            new_faces.append((a, b, c))
            
        self.faces = new_faces
    
    def _build_graph(self) -> nx.Graph:
        """
        Build a NetworkX graph from the grid vertices and faces.

        Nodes in the graph correspond to vertices, and edges are added between vertices that are 
        connected in any face. Additionally, each node stores its Cartesian coordinates and 
        computed latitude/longitude.

        Returns:
            nx.Graph: A graph representation of the icosahedral grid.
        """
        G = nx.Graph()
        for i, vertex in enumerate(self.vertices):
            # Convert the vertex to (lat, lon) and store it along with the position
            G.add_node(i, pos=vertex, lat_lon=self._cartesian_to_latlon(vertex))
        for face in self.faces:
            i, j, k = face
            G.add_edge(i, j)
            G.add_edge(j, k)
            G.add_edge(k, i)
        return G
    
    def _cartesian_to_latlon(self, xyz: np.ndarray) -> Tuple[float, float]:
        """
        Convert 3D Cartesian coordinates to latitude and longitude in degrees.

        Uses the arcsine of the z-coordinate for latitude and arctan2 for longitude.

        Args:
            xyz (np.ndarray): A 3-element array representing a point in 3D space.

        Returns:
            Tuple[float, float]: Latitude and longitude in degrees.
        """
        x, y, z = xyz
        lat = np.degrees(np.arcsin(z))
        lon = np.degrees(np.arctan2(y, x))
        return lat, lon
    
    def get_graph(self) -> nx.Graph:
        """
        Get the NetworkX graph representation of the icosahedral grid.

        Returns:
            nx.Graph: The graph constructed from the grid's vertices and faces.
        """
        return self.graph
    
    def get_vertices(self) -> np.ndarray:
        """
        Get the vertices of the icosahedral grid.

        Returns:
            np.ndarray: An array of shape (N, 3) containing the Cartesian coordinates of the vertices.
        """
        return self.vertices
    
    def get_latlon_coordinates(self) -> List[Tuple[float, float]]:
        """
        Get the geographic coordinates (latitude, longitude) of all vertices.

        Returns:
            List[Tuple[float, float]]: A list where each element is a tuple (lat, lon) corresponding to a vertex.
        """
        return [self._cartesian_to_latlon(v) for v in self.vertices]
    
    def get_adjacency_list(self) -> Dict[int, List[int]]:
        """
        Get the adjacency list representation of the grid's graph.

        Returns:
            Dict[int, List[int]]: A dictionary where each key is a vertex index and the corresponding value 
                                  is a list of adjacent vertex indices.
        """
        return {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()}

def create_icosahedral_graph(level: int = 3) -> nx.Graph:
    """
    Convenience function to create an icosahedral grid graph with a given subdivision level.

    Args:
        level (int): The subdivision level to apply. Higher values yield finer grids.

    Returns:
        nx.Graph: A NetworkX graph representing the icosahedral grid.
    """
    grid = IcosahedralGrid(subdivision_level=level)
    return grid.get_graph()

def get_grid_metadata(graph: nx.Graph) -> Dict:
    """
    Extract metadata from an icosahedral grid graph.

    The metadata includes the number of vertices, number of edges, positions of vertices, 
    their latitude/longitude pairs, and the list of edges.

    Args:
        graph (nx.Graph): A NetworkX graph representing the icosahedral grid.

    Returns:
        Dict: A dictionary containing metadata about the grid with the following keys:
              - 'num_vertices': Number of vertices in the graph.
              - 'num_edges': Number of edges in the graph.
              - 'positions': A NumPy array of vertex positions.
              - 'latlon': A list of (lat, lon) tuples for each vertex.
              - 'edges': A list of edges (tuples of vertex indices).
    """
    positions = np.array([data['pos'] for _, data in graph.nodes(data=True)])
    latlon = [data['lat_lon'] for _, data in graph.nodes(data=True)]
    edges = list(graph.edges())
    
    return {
        'num_vertices': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'positions': positions,
        'latlon': latlon,
        'edges': edges
    }
