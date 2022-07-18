# TODO: replace this with our own dask_utils
import ctypes

import dask
from dask.distributed import Client, LocalCluster
from graph_weather.utils.config import YAMLConfig


def __trim_dask_worker_memory() -> int:
    """
    Manually trim Dask worker memory. This will forcefully release allocated but unutilized memory.
    This may help reduce total memory used per worker when we operate on large numbers of small Python objects
    (e.g. non-numpy data and small numpy chunks)
    See:
        https://distributed.dask.org/en/stable/worker-memory.html
    and
        https://coiled.io/blog/tackling-unmanaged-memory-with-dask/
    """
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def init_dask(config: YAMLConfig):
    # NB: change this to point to a dir where you can write to!
    dask.config.set({"temporary_directory": config["model:dask:temp-dir"]})
    # forward port 9988 to access the dask dashboard
    cluster = LocalCluster(n_workers=16, threads_per_worker=2, dashboard_address=":9988")
    client = Client(cluster)
    client.run(__trim_dask_worker_memory)
    return client
