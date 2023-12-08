"""
Most of this is copied from https://github.com/NVIDIA/modulus/blob/main/modulus/utils/graphcast/graph_utils.py

There are some adaptions for PyTorch Geometric data structures instead.

# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

import torch
from torch import Tensor, testing
import numpy as np
from torch_geometric.data import Data, HeteroData


def latlon2xyz(latlon: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """
    Converts latlon in degrees to xyz
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.

    Parameters
    ----------
    latlon : Tensor
        Tensor of shape (N, 2) containing latitudes and longitudes
    radius : float, optional
        Radius of the sphere, by default 1
    unit : str, optional
        Unit of the latlon, by default "deg"

    Returns
    -------
    Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    """
    if unit == "deg":
        latlon = deg2rad(latlon)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")
    lat, lon = latlon[:, 0], latlon[:, 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack((x, y, z), dim=1)


def xyz2latlon(xyz: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """
    Converts xyz to latlon in degrees
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.

    Parameters
    ----------
    xyz : Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    radius : float, optional
        Radius of the sphere, by default 1
    unit : str, optional
        Unit of the latlon, by default "deg"

    Returns
    -------
    Tensor
        Tensor of shape (N, 2) containing latitudes and longitudes
    """
    lat = torch.arcsin(xyz[:, 2] / radius)
    lon = torch.arctan2(xyz[:, 1], xyz[:, 0])
    if unit == "deg":
        return torch.stack((rad2deg(lat), rad2deg(lon)), dim=1)
    elif unit == "rad":
        return torch.stack((lat, lon), dim=1)
    else:
        raise ValueError("Not a valid unit")


def get_edge_len(edge_src: Tensor, edge_dst: Tensor, axis: int = 1):
    """returns the length of the edge

    Parameters
    ----------
    edge_src : Tensor
        Tensor of shape (N, 3) containing the source of the edge
    edge_dst : Tensor
        Tensor of shape (N, 3) containing the destination of the edge
    axis : int, optional
        Axis along which the norm is computed, by default 1

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the length of the edge
    """
    return np.linalg.norm(edge_src - edge_dst, axis=axis)


def deg2rad(deg: Tensor) -> Tensor:
    """Converts degrees to radians

    Parameters
    ----------
    deg :
        Tensor of shape (N, ) containing the degrees

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the radians
    """
    return deg * np.pi / 180


def rad2deg(rad):
    """Converts radians to degrees

    Parameters
    ----------
    rad :
        Tensor of shape (N, ) containing the radians

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the degrees
    """
    return rad * 180 / np.pi


def azimuthal_angle(lon: Tensor) -> Tensor:
    """
    Gives the azimuthal angle of a point on the sphere

    Parameters
    ----------
    lon : Tensor
        Tensor of shape (N, ) containing the longitude of the point

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the azimuthal angle
    """
    angle = torch.where(lon >= 0.0, 2 * np.pi - lon, -lon)
    return angle


def polar_angle(lat: Tensor) -> Tensor:
    """
    Gives the polar angle of a point on the sphere

    Parameters
    ----------
    lat : Tensor
        Tensor of shape (N, ) containing the latitude of the point

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the polar angle
    """
    angle = torch.where(lat >= 0.0, lat, 2 * np.pi + lat)
    return angle


def geospatial_rotation(invar: Tensor, theta: Tensor, axis: str, unit: str = "rad") -> Tensor:
    """Rotation using right hand rule

    Parameters
    ----------
    invar : Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    theta : Tensor
        Tensor of shape (N, ) containing the rotation angle
    axis : str
        Axis of rotation
    unit : str, optional
        Unit of the theta, by default "rad"

    Returns
    -------
    Tensor
        Tensor of shape (N, 3) containing the rotated x, y, z coordinates
    """

    # get the right unit
    if unit == "deg":
        invar = rad2deg(invar)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")

    invar = torch.unsqueeze(invar, -1)
    rotation = torch.zeros((theta.size(0), 3, 3))
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    if axis == "x":
        rotation[:, 0, 0] += 1.0
        rotation[:, 1, 1] += cos
        rotation[:, 1, 2] -= sin
        rotation[:, 2, 1] += sin
        rotation[:, 2, 2] += cos
    elif axis == "y":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 2] += sin
        rotation[:, 1, 1] += 1.0
        rotation[:, 2, 0] -= sin
        rotation[:, 2, 2] += cos
    elif axis == "z":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 1] -= sin
        rotation[:, 1, 0] += sin
        rotation[:, 1, 1] += cos
        rotation[:, 2, 2] += 1.0
    else:
        raise ValueError("Invalid axis")

    outvar = torch.matmul(rotation, invar)
    outvar = outvar.squeeze()
    return outvar


def add_edge_features(graph: Data, pos: Tensor, normalize: bool = True) -> Data:
    """Adds edge features to the graph.

    Parameters
    ----------
    graph : Data
        The graph to add edge features to.
    pos : Tensor
        The node positions.
    normalize : bool, optional
        Whether to normalize the edge features, by default True

    Returns
    -------
    DGLGraph
        The graph with edge features.
    """

    if isinstance(pos, tuple):
        src_pos, dst_pos = pos
    else:
        src_pos = dst_pos = pos
    src, dst = graph.edge_index

    src_pos, dst_pos = src_pos[src.long()], dst_pos[dst.long()]
    dst_latlon = xyz2latlon(dst_pos, unit="rad")
    dst_lat, dst_lon = dst_latlon[:, 0], dst_latlon[:, 1]

    # azimuthal & polar rotation
    theta_azimuthal = azimuthal_angle(dst_lon)
    theta_polar = polar_angle(dst_lat)

    src_pos = geospatial_rotation(src_pos, theta=theta_azimuthal, axis="z", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_azimuthal, axis="z", unit="rad")
    # y values should be zero
    try:
        testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
    except ValueError:
        raise ValueError("Invalid projection of edge nodes to local ccordinate system")
    src_pos = geospatial_rotation(src_pos, theta=theta_polar, axis="y", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_polar, axis="y", unit="rad")
    # x values should be one, y & z values should be zero
    try:
        testing.assert_close(dst_pos[:, 0], torch.ones_like(dst_pos[:, 0]))
        testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
        testing.assert_close(dst_pos[:, 2], torch.zeros_like(dst_pos[:, 2]))
    except ValueError:
        raise ValueError("Invalid projection of edge nodes to local ccordinate system")

    # prepare edge features
    disp = src_pos - dst_pos
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)

    # normalize using the longest edge
    if normalize:
        max_disp_norm = torch.max(disp_norm)
        graph["edge_attr"] = torch.cat((disp / max_disp_norm, disp_norm / max_disp_norm), dim=-1)
    else:
        graph["edge_attr"] = torch.cat((disp, disp_norm), dim=-1)
    return graph


def add_node_features(graph: Data, pos: Tensor) -> Data:
    """Adds cosine of latitude, sine and cosine of longitude as the node features
    to the graph.

    Parameters
    ----------
    graph : DGLGraph
        The graph to add node features to.
    pos : Tensor
        The node positions.

    Returns
    -------
    graph : DGLGraph
        The graph with node features.
    """
    latlon = xyz2latlon(pos)
    lat, lon = latlon[:, 0], latlon[:, 1]
    graph["x"] = torch.stack((torch.cos(lat), torch.sin(lon), torch.cos(lon)), dim=-1)
    return graph


def plot_graph(graph: Data, **kwargs):
    """Plots the graph.

    Parameters
    ----------
    graph : Data
        The graph to plot.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx

    G = to_networkx(graph, to_undirected=True)
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 3d spring layout
    fixed_positions = {v: graph.pos[v] for v in range(len(graph.pos))}
    fixed_nodes = fixed_positions.keys()
    pos = nx.spring_layout(G, dim=3, fixed=fixed_nodes, pos=fixed_positions)
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=10, ec="w")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    _format_axes(ax)
    fig.tight_layout()
    # rotate the axes and update
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(0.01)
    plt.show()
