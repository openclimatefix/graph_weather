"""Utils for batching graphs."""

import torch


def batch(senders, edge_index, edge_attr=None, batch_size=1):
    """Build big batched graph.

    Returns nodes and edges of a big graph with batch_size disconnected copies of the original
    graph, with features shape [(b n) f].

    Args:
        senders (torch.Tensor): nodes' features.
        edge_index (torch.Tensor): edge index tensor.
        edge_attr (torch.Tensor, optional): edge attributes tensor, if None returns None.
            Defaults to None.
        batch_size (int): batch size. Defaults to 1.

    Returns:
        batched_senders, batched_edge_index, batched_edge_attr
    """
    ns = senders.shape[0]
    batched_senders = senders
    batched_edge_attr = edge_attr
    batched_edge_index = edge_index

    for i in range(1, batch_size):
        batched_senders = torch.cat([batched_senders, senders], dim=0)
        batched_edge_index = torch.cat([batched_edge_index, edge_index + i * ns], dim=1)

        if edge_attr is not None:
            batched_edge_attr = torch.cat([batched_edge_attr, edge_attr], dim=0)

    return batched_senders, batched_edge_index, batched_edge_attr


def hetero_batch(senders, receivers, edge_index, edge_attr=None, batch_size=1):
    """Build big batched heterogenous graph.

    Returns nodes and edges of a big graph with batch_size disconnected copies of the original
    graph, with features shape [(b n) f].

    Args:
        senders (torch.Tensor): senders' features.
        receivers (torch.Tensor): receivers' features.
        edge_index (torch.Tensor): edge index tensor.
        edge_attr (torch.Tensor, optional): edge attributes tensor, if None returns None.
            Defaults to None.
        batch_size (int): batch size. Defaults to 1.

    Returns:
        batched_senders, batched_edge_index, batched_edge_attr
    """
    ns = senders.shape[0]
    nr = receivers.shape[0]
    nodes_shape = torch.tensor([[ns], [nr]]).to(edge_index)
    batched_senders = senders
    batched_receivers = receivers
    batched_edge_attr = edge_attr
    batched_edge_index = edge_index

    for i in range(1, batch_size):
        batched_senders = torch.cat([batched_senders, senders], dim=0)
        batched_receivers = torch.cat([batched_receivers, receivers], dim=0)
        batched_edge_index = torch.cat([batched_edge_index, edge_index + i * nodes_shape], dim=1)
        if edge_attr is not None:
            batched_edge_attr = torch.cat([batched_edge_attr, edge_attr], dim=0)

    return batched_senders, batched_receivers, batched_edge_index, batched_edge_attr
