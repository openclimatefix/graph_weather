import random
import time
from torch.utils.data import Subset


def get_all_val_attrs(obj):
    def _good(attr):
        return (not attr.startswith('__')) and (not callable(getattr(obj, attr)))

    attrs = [attr for attr in dir(obj) if _good(attr)]

    return attrs


def get_subset(dataset, ratio, attributes=None):
    length = len(dataset)
    subset = Subset(dataset, indices=random.sample(range(length), int(ratio * length)))

    # original dataset might have some special attributes
    # subset_attrs = get_all_val_attrs(subset)
    # og_attrs = get_all_val_attrs(dataset)

    # for attr in og_attrs:
    #    if attr == 'indices':
    #        raise ValueError("og_attrs can't have 'indices'")
    #    if attr not in subset_attrs:
    #        setattr(subset, attr, getattr(dataset, attr))

    if attributes:
        for attr in attributes:
            assert hasattr(dataset, attr)
            setattr(subset, attr, getattr(dataset, attr))

    return subset

def get_delta_minute(start_time):
    # start_time in seconds
    return (time.time() - start_time) / 60.

