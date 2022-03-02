import numpy as np
from numpy import concatenate, cumsum, diff, int64, isin, ones, squeeze, tile, where, zeros
from numpy.random import normal
from torch import tensor
from torch.nn.functional import one_hot


def process_node_window(
    node_data, node_types, apply_onehot=False, onehot_classes=0, inlet_velocity=None
):
    """
    Concatenates with node features with one-hot encoding of node types

    node_data: time x node x feature array
    node_types: node x one-hot dim array
    apply_onehot: boolean
    onehot_classes: integer
    inlet_velocity: float; used as a node feature for now; to do: replace with node x additional_features array
    """

    num_nodes = node_data.shape[1]

    node_types_ = (
        one_hot(tensor(node_types.astype(int64).flatten()), onehot_classes).numpy()
        if apply_onehot
        else node_types
    )

    node_data = [node_data.transpose(1, 0, 2).reshape((num_nodes, -1)), node_types_]
    if inlet_velocity is not None:
        node_data.append(inlet_velocity * ones((num_nodes, 1)))

    return concatenate(node_data, axis=-1)


def get_sample(dataset, source_node_idx, time_idx, num_prev_velocities=5, noise_sd=None):
    """
    Returns input position and
    Args:
        dataset:
        source_node_idx:
        time_idx:
        num_prev_velocities:
        noise_sd:

    Returns:

    """
    """
    Returns input position and velocity data window (with noise) and output acceleration

    dataset: time x node x feature numpy array
    source_node_idx: source node indices
    num_prev_velocities: number of previous velocities to include as input
    noise_sd: noise standard deviation
    """

    position_data = dataset[time_idx : (time_idx + num_prev_velocities + 1), :, :]
    next_position = dataset[[time_idx + num_prev_velocities + 1], :, :]

    # add noise
    if noise_sd is not None:
        # noise_sd = noise_sd / (len(prev_velocities) ** 0.5)
        noise = tile(noise_sd, (num_prev_velocities, dataset.shape[1], 1))
        noise = normal(0, noise)
        c_noise = cumsum(
            noise, axis=0
        )  # first cumsum for random walk; second for euler integration
        c_noise = concatenate(
            [zeros((1, c_noise.shape[1], c_noise.shape[2])), cumsum(c_noise, axis=0)], axis=0
        )

        position_data += c_noise

    velocity_data = diff(position_data, axis=0)
    last_position = dataset[
        [time_idx + num_prev_velocities],
        :,
        :,
    ].copy()
    node_data = np.concatenate([last_position, velocity_data], axis=0)

    # output acceleration
    outputs = np.concatenate([position_data[-2:], next_position], axis=0)
    outputs = squeeze(diff(diff(outputs, axis=0), axis=0), axis=0)

    # to consider:
    # #for sources, shift window ahead by 1 timepoint, do not add noise, and set output to 0
    # node_data[:, source_node_idx, :] = dataset[(time_idx+1):(time_idx+window_length+1), source_node_idx,:].copy()
    # outputs[source_node_idx, :] = 0

    return node_data, outputs
