"""Model for forecasting weather from NWP states"""


from huggingface_hub import PyTorchModelHubMixin
from numpy import array, concatenate, expand_dims, float32
from torch import from_numpy, no_grad
from torch_geometric.data import Data

from .gnn import GNN
from .utils.mgn_utils import process_node_window
from .utils.Normalizer import Normalizer


class GraphWeatherForecaster(GNN, PyTorchModelHubMixin):
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
        # adaptive mesh
        use_adaptive_mesh=False,
        **kwargs
    ):

        """
        MeshGraphNets model (arXiv:2010.03409); default values are based on the paper/supplement

        Originally from https://github.com/CCSI-Toolset/MGN, US Government license

        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        out_dim: output dimension
        out_dim_node: encoded node feature dimension
        out_dim_edge: encoded edge feature dimension
        hidden_dim_node: node encoder MLP dimension
        hidden_dim_edge: edge encoder MLP dimension
        hidden_layers_node: number of node encoder MLP layers
        hidden_layers_edge: number of edge encoder MLP layers
        mp_iterations: number of message passing iterations
        hidden_dim_processor_node: MGN node processor MLP dimension
        hidden_dim_processor_edge: MGN edge processor MLP dimension
        hidden_layers_processor_node: number of MGN node processor MLP layers
        hidden_layers_processor_edge: number of MGN edge processor MLP layers
        mlp_norm_type: MLP normalization type ('LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm')
        hidden_dim_decoder: decoder MLP dimension
        hidden_layers_decoder: decoder MLP layers
        output_type: output type ('state', 'velocity', 'acceleration')
        use_adaptive_mesh: if True, use adaptive; not functional yet
        """

        super().__init__(
            in_dim_node,
            in_dim_edge,
            out_dim,
            out_dim_node,
            out_dim_edge,
            hidden_dim_node,
            hidden_dim_edge,
            hidden_layers_node,
            hidden_layers_edge,
            mp_iterations,
            hidden_dim_processor_node,
            hidden_dim_processor_edge,
            hidden_layers_processor_node,
            hidden_layers_processor_edge,
            mlp_norm_type,
            hidden_dim_decoder,
            hidden_layers_decoder,
            output_type,
            **kwargs
        )

        self.in_dim_node = in_dim_node
        self.in_dim_edge = in_dim_edge
        self.out_dim = out_dim

        self.use_adaptive_mesh = use_adaptive_mesh

        self._node_normalizer = Normalizer(size=in_dim_node)
        self._edge_normalizer = Normalizer(size=in_dim_edge)
        self._output_normalizer = Normalizer(size=out_dim)

        # if self.use_adaptive_mesh:
        #     #if true, assume edge_attr corresponds to the 'world' edges (fixed)
        #     #and edge_attr_adaptive corresponds to the 'mesh' edges (adaptive)
        #     #future work: adapt to graphs at multiple resolutions

        #     from .GNNComponents.GNNComponents import MLP
        #     from .GNNComponents.MultiGNNComponents import GraphProcessor

        #     self.edge_encoder_adaptive = MLP(in_dim_edge, out_dim_edge, hidden_dim_edge, hidden_layers_edge)
        #     self.graph_processor = GraphProcessor(mp_iterations, 2,
        #         out_dim_node, out_dim_edge,
        #         hidden_dim_processor_node, hidden_dim_processor_edge,
        #         hidden_layers_processor_node, hidden_layers_processor_edge)

    def forward(self, graph):

        # encode node/edge features
        out = self.node_encoder(self._node_normalizer(graph.x, accumulate=self.training))
        edge_attr = self.edge_encoder(
            self._edge_normalizer(graph.edge_attr, accumulate=self.training)
        )

        # if self.use_adaptive_mesh:
        #     edge_attr_adaptive = self.edge_encoder_adaptive(graph.edge_attr_adaptive)
        #     out, _ = self.graph_processor(out,
        #         [graph.edge_index, graph.edge_index_adaptive],
        #         [edge_attr, edge_attr_adaptive])
        # else:
        #     out, _ = self.graph_processor(out, graph.edge_index, edge_attr)

        # message passing
        out, _ = self.graph_processor(out, graph.edge_index, edge_attr)

        # decode
        out = self.node_decoder(out)

        return out

    # default update/rollout functions
    # extend class/override these functions if we want to update states differently
    # or specify them in the dataset classes
    def update_function(
        self, mgn_output_np, current_state=None, previous_state=None, source_data=None
    ):

        """
        Default state update function;
        Extend and override this function, or add as a dataset class attribute

        mgn_output_np: MGN output
        current_state: Current state
        previous_state: Previous state (for acceleration-based updates)
        source_data: Source/scripted node data
        """

        with no_grad():
            if self.mgn_output_type == "acceleration":
                assert current_state is not None
                assert previous_state is not None
                next_state = 2 * current_state - previous_state + mgn_output_np
            elif self.mgn_output_type == "velocity":
                assert current_state is not None
                next_state = current_state + mgn_output_np
            else:  # state
                next_state = mgn_output_np.copy()

            if type(source_data) is dict:
                for key in source_data:
                    next_state[key] = source_data[key]
            elif type(source_data) is tuple:
                next_state[source_data[0]] = source_data[1]
            # else: warning?

        return next_state

    def rollout(
        self,
        device,
        initial_window_data,
        graph,
        node_types,
        node_coordinates,
        onehot_info=None,
        inlet_velocity=None,
        rollout_iterations=1,
        source_data=None,
        update_function=None,
    ):  # if None, use the default update_function (above; uses same variables for input and output)

        """
        graph: torch_geometric.data.Data object with the following attributes (see PNNLGraphDataset.py for graph construction):
               x: node x feature array (volume fraction, pressure, node type, inlet velocity, etc.)
               edge_index: 2 x edge array
               edge_attr: edge x feature matrix (distance, relative coordinates)
        initial_window_data: window x nodes x features array; assume window with size >= 2
        node_types: node x one-hot dim array
        node_coordinates: node x dimension array
        inlet_velocity: float
        rollout_iterations: int; number of timepoints to simulate
        source_data: dict({node id: time x features np array}) or tuple (node ids, node x time x feature np array); simulated nodes/boundary conditions
        """

        assert (self.output_type != "acceleration") or (initial_window_data.shape[-1] >= 2)

        input_len = initial_window_data.shape[-1]

        self.eval()

        update_func = self.update_function if update_function is None else update_function

        rollout_output = []
        with no_grad():
            current_window_data = initial_window_data.copy()
            for i in range(rollout_iterations):

                if type(source_data) is dict:
                    source_data_i = concatenate(
                        [source_data[id][i, :] for id in source_data.keys()]
                    )
                    source_data_i = (list(source_data.keys()), source_data_i)
                elif type(source_data) is tuple:
                    source_data_i = (source_data[0], source_data[1][i, :, :])
                else:
                    source_data_i = None

                # Current version: replace input source data with data one timestep ahead
                # to do: add a config option to turn this off here and in mgn_utils.py
                if source_data_i is not None:
                    current_window_data[:-1, source_data_i[0]] = current_window_data[
                        1:, source_data_i[0]
                    ]
                    current_window_data[-1, source_data_i[0]] = source_data_i[1]

                # process/reshape data
                node_data = from_numpy(
                    process_node_window(
                        current_window_data,
                        node_coordinates,
                        node_types,
                        onehot_info[0],
                        onehot_info[1],
                        inlet_velocity,
                    ).astype(float32)
                )

                # apply MGN model
                input_graph = Data(
                    x=node_data, edge_index=graph.edge_index, edge_attr=graph.edge_attr
                ).to(device)
                # mgn_output = self.forward(input_graph).cpu().numpy()
                mgn_output = (
                    self._output_normalizer.inverse(self.forward(input_graph)).cpu().numpy()
                )

                # #to do: adaptive meshing
                # if self.use_adaptive_mesh:
                #     pass

                # update state
                next_state = update_func(
                    mgn_output,
                    self.output_type,
                    current_window_data[-1],
                    current_window_data[-2] if len(current_window_data) > 1 else None,
                    source_data_i if source_data is not None else None,
                )
                current_window_data = concatenate(
                    [current_window_data[1:], expand_dims(next_state[:, :input_len], 0)], axis=0
                )

                rollout_output.append(next_state)

        return array(rollout_output)
