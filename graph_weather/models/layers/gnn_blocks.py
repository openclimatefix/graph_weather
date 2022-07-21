"""
Functions for building GNNs.

This code is taken from https://github.com/CCSI-Toolset/MGN
which is available under the US Government License.

Copyright and License
=====================

Copyright (c) 2012 - 2022

Copyright Notice
----------------

MGN was produced under the DOE Carbon Capture Simulation Initiative (CCSI),
and is copyright (c) 2012 - 2022 by the software owners: Oak Ridge Institute
for Science and Education (ORISE), TRIAD National Security, LLC., Lawrence
Livermore National Security, LLC., The Regents of the University of California,
through Lawrence Berkeley National Laboratory, Battelle Memorial Institute,
Pacific Northwest Division through Pacific Northwest National Laboratory,
Carnegie Mellon University, West Virginia University, Boston University,
the Trustees of Princeton University, The University of Texas at Austin,
URS Energy & Construction, Inc., et al..  All rights reserved.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.
As such, the U.S. Government has been granted for itself and others acting
on its behalf a paid-up, nonexclusive, irrevocable, worldwide license 
in the Software to reproduce, distribute copies to the public, prepare 
derivative works, and perform publicly and display publicly, and to permit 
other to do so.

License Agreement
-----------------

MGN Copyright (c) 2012 - 2022, by the software owners: Oak Ridge Institute for 
Science and Education (ORISE), TRIAD National Security, LLC., Lawrence Livermore 
National Security, LLC., The Regents of the University of California, through 
Lawrence Berkeley National Laboratory, Battelle Memorial Institute, Pacific 
Northwest Division through Pacific Northwest National Laboratory, Carnegie 
Mellon University, West Virginia University, Boston University, the Trustees 
of Princeton University, The University of Texas at Austin, URS Energy & 
Construction, Inc.,  et al.  All rights reserved.


Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the Carbon Capture Simulation Initiative, U.S. Dept. 
of Energy, the National Energy Technology Laboratory, Oak Ridge Institute 
for Science and Education (ORISE), TRIAD National Security, LLC.,  Lawrence 
Livermore National Security, LLC., the University of California, Lawrence 
Berkeley National Laboratory, Battelle Memorial Institute, Pacific Northwest 
National Laboratory, Carnegie Mellon University, West Virginia University, 
Boston University, the Trustees of Princeton University, the University of 
Texas at Austin, URS Energy & Construction, Inc.,  nor the names of its 
contributors may be used to endorse or promote products derived from this 
software without specific prior written permission.

 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.


You are under no obligation whatsoever to provide any bug fixes, patches, 
or upgrades to the features, functionality or performance of the source 
code ("Enhancements") to anyone; however, if you choose to make your 
Enhancements available either publicly, or directly to Lawrence Berkeley 
National Laboratory, without imposing a separate written license agreement 
for such Enhancements, then you hereby grant the following license: a 
non-exclusive, royalty-free perpetual license to install, use, modify, 
prepare derivative works, incorporate into other computer software, 
distribute, and sublicense such enhancements or derivative works 
thereof, in binary and source code form.
"""
from typing import Tuple, Optional

import torch
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum


class MLP(nn.Module):
    """MLP for graph processing"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: Optional[str] = "LayerNorm",
    ) -> None:
        """
        MLP

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type one of 'LayerNorm', 'GraphNorm',
                'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(MLP, self).__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the MLP

        Args:
            x: Node or edge features

        Returns:
            The transformed tensor
        """
        return self.model(x)


#############################
# issue with MessagePassing class:
# Only node features are updated after MP iterations
# Need to use MetaLayer to also allow edge features to update


class EdgeProcessor(nn.Module):
    """EdgeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: Optional[str] = "LayerNorm",
    ) -> None:
        """
        Edge processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layers
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(EdgeProcessor, self).__init__()
        self.edge_mlp = MLP(2 * in_dim_node + in_dim_edge, in_dim_edge, hidden_dim, hidden_layers, norm_type)

    def forward(
        self,
        src: torch.Tensor,
        dest: torch.Tensor,
        edge_attr: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the edge part of the message passing

        Args:
            src: Source node tensor
            dest: Destination node tensor
            edge_attr: Edge attributes
            u: Global attributes, ignored
            batch: Batch Ids, ignored

        Returns:
            The updated edge attributes
        """
        del u, batch  # not used
        out = torch.cat([src, dest, edge_attr], -1)  # concatenate source node, destination node, and edge embeddings
        out = self.edge_mlp(out)
        out += edge_attr  # residual connection
        return out


class NodeProcessor(nn.Module):
    """NodeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: Optional[str] = "LayerNorm",
    ) -> None:
        """
        Node Processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(NodeProcessor, self).__init__()
        self.node_mlp = MLP(in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the node feature updates in message passing

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes
            u: Global attributes, ignored
            batch: Batch IDX, ignored

        Returns:
            torch.Tensor with updated node attributes
        """
        del u, batch  # not used
        _, col = edge_index
        out = scatter_sum(edge_attr, col, dim=0)  # aggregate edge message by target
        out = torch.cat([x, out], dim=-1)
        out = self.node_mlp(out)
        out += x  # residual connection
        return out


def build_graph_processor_block(
    in_dim_node: int = 128,
    in_dim_edge: int = 128,
    hidden_dim_node: int = 128,
    hidden_dim_edge: int = 128,
    hidden_layers_node: int = 2,
    hidden_layers_edge: int = 2,
    norm_type: Optional[str] = "LayerNorm",
) -> nn.Module:
    """
    Build the Graph Net Block

    Args:
        in_dim_node: Input node feature dimension
        in_dim_edge: Input edge feature dimension
        hidden_dim_node: Number of nodes in hidden layer for graph node processing
        hidden_dim_edge: Number of nodes in hidden layer for graph edge processing
        hidden_layers_node: Number of hidden layers for node processing
        hidden_layers_edge: Number of hidden layers for edge processing
        norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
    Returns:
        nn.Module for the graph processing block
    """

    return MetaLayer(
        edge_model=EdgeProcessor(in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type),
        node_model=NodeProcessor(in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type),
    )


class GraphProcessor(nn.Module):
    """Overall graph processor"""

    def __init__(
        self,
        mp_iterations: int = 15,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim_node: int = 128,
        hidden_dim_edge: int = 128,
        hidden_layers_node: int = 2,
        hidden_layers_edge: int = 2,
        norm_type: Optional[str] = "LayerNorm",
    ) -> None:
        """
        Graph Processor

        Args:
            mp_iterations: number of message-passing iterations (graph processor blocks)
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim_node: Number of nodes in hidden layers for node processing
            hidden_dim_edge: Number of nodes in hidden layers for edge processing
            hidden_layers_node: Number of hidden layers for node processing
            hidden_layers_edge: Number of hidden layers for edge processing
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(GraphProcessor, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(mp_iterations):
            self.blocks.append(
                build_graph_processor_block(
                    in_dim_node,
                    in_dim_edge,
                    hidden_dim_node,
                    hidden_dim_edge,
                    hidden_layers_node,
                    hidden_layers_edge,
                    norm_type,
                )
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute updates to the graph in message passing method

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes

        Returns:
            Updated nodes and edge attributes
        """
        for block in self.blocks:
            x, edge_attr, _ = block(x, edge_index, edge_attr)

        return x, edge_attr
