from torch import cat
from torch.nn import Module, ModuleList
from torch_scatter import scatter_sum

from .graph_net_block import MLP, EdgeProcessor


class MetaLayerMultigraph(Module):
    # Based on torch_geometric.nn.MetaLayer; needed for graphs with multiple edge types
    # Not using global_model for now

    def __init__(self, edge_models=None, node_model=None):

        """
        MetaLayer for multigraphs

        edge_models: list of edge processor models
        node_model: node_model
        """

        super(MetaLayerMultigraph, self).__init__()
        self.edge_models = edge_models
        self.node_model = node_model

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.node_model, "reset_parameters"):
            self.node_model.reset_parameters()

        for item in self.edge_models:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x, edge_indices, edge_attrs=None, u=None, batch=None):

        if self.edge_models is not None:
            for i, (em, ei, ea) in enumerate(zip(self.edge_models, edge_indices, edge_attrs)):
                edge_attrs[i] = em(
                    x[ei[0]], x[ei[1]], ea, u, batch if batch is None else batch[row]
                )

        if self.node_model is not None:
            x = self.node_model(x, edge_indices, edge_attrs, u, batch)

        return x, edge_attrs, u


class NodeProcessor(Module):
    def __init__(
        self,
        in_dim_node=128,
        in_dim_edge=128,
        hidden_dim=128,
        hidden_layers=2,
        norm_type="LayerNorm",
    ):

        """
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension (for now, assume to be the same for all edge models)
        hidden_dim: number of nodes in a hidden layer; future work: accept integer array
        hidden_layers: number of hidden layers
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        # super(NodeProcessor, self).__init__()
        self.node_mlp = MLP(
            in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type
        )

    def forward(self, x, edge_indices, edge_attrs, u=None, batch=None):

        out = [x]
        for ei, ea in zip(edge_indices, edge_attrs):
            out.append(scatter_sum(ea, ei[1], dim=0))

        out = cat(out, dim=-1)
        out = self.node_mlp(out)
        out += x  # residual connection

        return out


def build_graph_processor_block(
    num_edge_models=1,
    in_dim_node=128,
    in_dim_edge=128,
    hidden_dim_node=128,
    hidden_dim_edge=128,
    hidden_layers_node=2,
    hidden_layers_edge=2,
    norm_type="LayerNorm",
):

    edge_models = [
        EdgeProcessor(in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type)
        for _ in range(num_edge_models)
    ]
    node_model = NodeProcessor(
        in_dim_node, in_dim_edge * num_edge_models, hidden_dim_node, hidden_layers_node, norm_type
    )

    return MetaLayerMultigraph(edge_models=edge_models, node_model=node_model)


class MultiGraphBlock(Module):
    def __init__(
        self,
        mp_iterations=15,
        num_edge_models=1,
        in_dim_node=128,
        in_dim_edge=128,
        hidden_dim_node=128,
        hidden_dim_edge=128,
        hidden_layers_node=2,
        hidden_layers_edge=2,
        norm_type="LayerNorm",
    ):

        """
        Graph processor

        mp_iterations: number of message-passing iterations (graph processor blocks)
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim_node: number of nodes in a hidden layer for graph node processing
        hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
        hidden_layers_node: number of hidden layers for graph node processing
        hidden_layers_edge: number of hidden layers for graph edge processing
        norm_type: normalization type; one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None

        """

        # super(GraphProcessor, self).__init__()

        self.blocks = ModuleList()
        for _ in range(mp_iterations):
            self.blocks.append(
                build_graph_processor_block(
                    num_edge_models,
                    in_dim_node,
                    in_dim_edge,
                    hidden_dim_node,
                    hidden_dim_edge,
                    hidden_layers_node,
                    hidden_layers_edge,
                    norm_type,
                )
            )

    def forward(self, x, edge_indices, edge_attrs):
        for block in self.blocks:
            x, edge_attrs, _ = block(x, edge_indices, edge_attrs)

        return x, edge_attrs
