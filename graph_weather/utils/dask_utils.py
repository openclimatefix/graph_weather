# TODO: replace this with our own dask_utils
import ctypes

import dask
from dask.distributed import Client, LocalCluster
from graph_weather.utils.config import YAMLConfig


def __trim_dask_worker_memory() -> int:
    """
    Manually trim Dask worker memory. This will forcefully release allocated but unutilized memory.
    This may help reduce total memory used per worker.
    See:
        https://distributed.dask.org/en/stable/worker-memory.html
    and
        https://coiled.io/blog/tackling-unmanaged-memory-with-dask/
    """
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def init_dask_cluster(config: YAMLConfig) -> LocalCluster:
    dask.config.set(
        {
            # temporary directory
            "temporary_directory": config["model:dask:temp-dir"],
            # this high initial guess tells the scheduler to spread tasks
            "distributed.scheduler.unknown-task-duration": "10s",
            # worker memory management
            "distributed.worker.memory.spill": 0.9,
            "distributed.worker.memory.target": 0.85,
            "distributed.worker.memory.pause": 0.95,
            "distributed.worker.memory.terminate": False,
        }
    )
    return LocalCluster(
        n_workers=config["model:dask:num-workers"],
        threads_per_worker=config["model:dask:num-threads-per-worker"],
        dashboard_address=f":{config['model:dask:dashboard-port']}",
        scheduler_port=config["model:dask:scheduler-port"],
    )


def init_dask_client(scheduler_addr: str, config: YAMLConfig) -> Client:
    client = Client(scheduler_addr)
    if config["model:dask:trim-worker-memory"]:
        client.run(__trim_dask_worker_memory)
    return client
