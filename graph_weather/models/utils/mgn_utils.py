
from torch import tensor
from torch.nn.functional import one_hot
from numpy import concatenate, ones, int64, squeeze, diff, tile, where, isin
from numpy.random import normal

def process_node_window(node_data, node_coordinates, node_types, 
        apply_onehot = False,
        onehot_classes = 0, 
        inlet_velocity=None):

    '''
    Concatenates with node features with one-hot encoding of node types

    node_data: time x node x feature array
    node_types: node x one-hot dim array
    node_coordinates: node x dimension array
    apply_onehot: boolean
    onehot_classes: integer
    #inlet_velocity: float; we'll use it as a node feature for now, unless there's a better place to use it

    '''

    num_nodes = node_data.shape[1]

    node_types_ = one_hot(tensor(node_types.astype(int64).flatten()), onehot_classes).numpy() if apply_onehot else node_types

    node_data = [node_data.transpose(1,0,2).reshape((num_nodes, -1)), node_coordinates, node_types_]
    if inlet_velocity is not None:
        node_data.append(inlet_velocity * ones((num_nodes, 1)))

    return concatenate(node_data, axis=-1)

def get_sample(dataset, source_node_idx,
        time_idx, 
        window_length=5,
        output_type='acceleration', 
        noise_sd=None,
        noise_gamma=0.1,
        shift_source=True):

    '''
    Returns position data window (with noise) and output velocity

    source_node_idx: source node indices
    time_idx: current time index  
    window_length: input window length
    output_type: output type; one of 'state', 'velocity', or 'acceleration'
    noise_sd: noise standard deviation
    noise_gamma: noise gamma (see noise details in arXiv:2010.03409)
    shift_source: if True, shift input source nodes ahead by one timestep; noise is not included
    '''

    node_data = dataset[time_idx:(time_idx+window_length),:,:].copy()

    #compute output
    if output_type == 'acceleration':
        outputs = dataset[(time_idx + window_length-2):(time_idx + window_length+1),:,:]
        outputs = squeeze(diff(diff(outputs, axis=0), axis=0), axis=0)
    elif output_type == 'velocity':
        outputs = dataset[(time_idx + window_length-1):(time_idx + window_length+1),:,:]
        outputs = squeeze(diff(outputs, axis=0), axis=0)
    else:
        outputs = dataset[time_idx+window_length,:,:].copy()

    #add noise to position and output
    if noise_sd is not None:
        noise = tile(noise_sd, (node_data.shape[1], 1))
        noise = normal(0, noise)
        #input noise
        node_data[-1] += noise
        #output adjustment
        if output_type == 'acceleration':
            #acceleration_p = x_{t+1} - 2 * x_{t} + x_{t-1} - 2 * noise
            #acceleration_v = x_{t+1} - 2 * x_{t} + x_t{-1} - noise
            #adjustment = 2 * gamma * noise + (1-gamma) * noise = noise * (1 + gamma)
            outputs -= (1 + noise_gamma) * noise
        elif output_type == 'velocity':
            #velocity_adj = x_{t+1} - (x_{t} + noise)
            outputs -= noise
        #else: nothing for state

    #for sources, shift window ahead by 1 timepoint, do not add noise
    #also need to update MeshGraphNets.py update_state and rollout functions (and loss?)
    #to do: add a config parameter to turn this off
    if shift_source:
        node_data[:, source_node_idx, :] = dataset[(time_idx+1):(time_idx+window_length+1), source_node_idx,:].copy()
    else:
        pass #no noise
        node_data[:, source_node_idx, :] = dataset[time_idx:(time_idx+window_length), source_node_idx,:].copy()

    #note: do not set source output to 0 (outputs[source_node_idx, :] = 0);
    #current version still affects training loss

    return node_data, outputs