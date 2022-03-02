"""Base model for both types of weather models"""
import collections
import functools

import torch

EdgeSet = collections.namedtuple("EdgeSet", ["name", "features", "senders", "receivers"])
MultiGraph = collections.namedtuple("Graph", ["node_features", "edge_sets"])


class GraphNetBlock(torch.nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, name="GraphNetBlock"):
        super(GraphNetBlock, self).__init__()
        self._model_fn = model_fn

    def _update_edge_features(self, node_features, edge_set):
        """Aggregrates node features, and applies edge function."""
        sender_features = tf.gather(node_features, edge_set.senders)
        receiver_features = tf.gather(node_features, edge_set.receivers)
        features = [sender_features, receiver_features, edge_set.features]
        with tf.variable_scope(edge_set.name + "_edge_fn"):
            return self._model_fn()(torch.concat(features, dim=-1))

    def _update_node_features(self, node_features, edge_sets):
        """Aggregrates edge features, and applies node function."""
        num_nodes = tf.shape(node_features)[0]
        features = [node_features]
        for edge_set in edge_sets:
            features.append(
                tf.math.unsorted_segment_sum(edge_set.features, edge_set.receivers, num_nodes)
            )
        with tf.variable_scope("node_fn"):
            return self._model_fn()(tf.concat(features, axis=-1))

    def _build(self, graph):
        """Applies GraphNetBlock and returns updated MultiGraph."""

        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features, edge_set)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        new_node_features = self._update_node_features(graph.node_features, new_edge_sets)

        # add residual connections
        new_node_features += graph.node_features
        new_edge_sets = [
            es._replace(features=es.features + old_es.features)
            for es, old_es in zip(new_edge_sets, graph.edge_sets)
        ]
        return MultiGraph(new_node_features, new_edge_sets)


class EncodeProcessDecode(torch.nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(
        self,
        output_size,
        latent_size,
        num_layers,
        message_passing_steps,
    ):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps

    def _make_mlp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers + [output_size]
        network = snt.nets.MLP(widths, activate_final=False)
        if layer_norm:
            network = torch.Sequential([network, torch.nn.LayerNorm()])
        return network

    def _encoder(self, graph):
        """Encodes node and edge features into latent features."""
        with tf.variable_scope("encoder"):
            node_latents = self._make_mlp(self._latent_size)(graph.node_features)
            new_edges_sets = []
            for edge_set in graph.edge_sets:
                latent = self._make_mlp(self._latent_size)(edge_set.features)
                new_edges_sets.append(edge_set._replace(features=latent))
        return MultiGraph(node_latents, new_edges_sets)

    def _decoder(self, graph):
        """Decodes node features from graph."""
        with tf.variable_scope("decoder"):
            decoder = self._make_mlp(self._output_size, layer_norm=False)
            return decoder(graph.node_features)

    def _build(self, graph):
        """Encodes and processes a multigraph, and returns node features."""
        model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
        latent_graph = self._encoder(graph)
        for _ in range(self._message_passing_steps):
            latent_graph = GraphNetBlock(model_fn)(latent_graph)
        return self._decoder(latent_graph)
