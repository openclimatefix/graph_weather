from torch.nn import Module

from .layers.graph_net_block import MLP, GraphProcessor


class GNN(Module):
    # default values based on MeshGraphNets paper/supplement
    def __init__(
        self,
        # data attributes:
        in_dim_node,  # includes data window, node type, inlet velocity
        in_dim_edge,  # distance and relative coordinates
        out_dim,  # includes x-velocity, y-velocity, volume fraction, pressure (or a subset)
        # encoding attributes:
        out_dim_node=128,
        out_dim_edge=128,
        hidden_dim_node=128,
        hidden_dim_edge=128,
        hidden_layers_node=2,
        hidden_layers_edge=2,
        # graph processor attributes:
        mp_iterations=15,
        hidden_dim_processor_node=128,
        hidden_dim_processor_edge=128,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        mlp_norm_type="LayerNorm",
        # decoder attributes:
        hidden_dim_decoder=128,
        hidden_layers_decoder=2,
        output_type="acceleration",
        # other:
        **kwargs
    ):

        super(GNN, self).__init__()

        self.node_encoder = MLP(
            in_dim_node, out_dim_node, hidden_dim_node, hidden_layers_node, mlp_norm_type
        )
        self.edge_encoder = MLP(
            in_dim_edge, out_dim_edge, hidden_dim_edge, hidden_layers_edge, mlp_norm_type
        )
        self.graph_processor = GraphProcessor(
            mp_iterations,
            out_dim_node,
            out_dim_edge,
            hidden_dim_processor_node,
            hidden_dim_processor_edge,
            hidden_layers_processor_node,
            hidden_layers_processor_edge,
            mlp_norm_type,
        )
        self.node_decoder = MLP(
            out_dim_node, out_dim, hidden_dim_decoder, hidden_layers_decoder, None
        )
        self.output_type = output_type

    # graph: torch_geometric.data.Data object with the following attributes:
    #       x: node x feature array (volume fraction, pressure, node type, inlet velocity, etc.)
    #       edge_index: 2 x edge array
    #       edge_attr: edge x feature matrix (distance, relative coordinates)

    def forward(self, graph):
        out = self.node_encoder(graph.x)
        edge_attr = self.edge_encoder(graph.edge_attr)
        out, _ = self.graph_processor(out, graph.edge_index, edge_attr)
        out = self.node_decoder(
            out
        )  # paper: corresponds to velocity or acceleration at this point; loss is based on one of these, not the actual state

        return out

    # # implement these in subclasses:

    # def update_state(mgn_output_np,
    #         current_state=None, previous_state=None,
    #         source_data=None):
    #     pass

    # def rollout(device,
    #         initial_window_data,
    #         graph,
    #         node_types,
    #         node_coordinates,
    #         onehot_info = None,
    #         inlet_velocity=None, rollout_iterations=1,
    #         source_data=None):
    #     pass
