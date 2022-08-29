import ctypes
import dask


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


def init_dask_config(temp_dir: str) -> None:
    dask.config.set(
        {
            # temporary directory
            "temporary_directory": temp_dir,
            # this high initial guess tells the scheduler to spread tasks
            # "distributed.scheduler.unknown-task-duration": "10s",
            # worker memory management
            "distributed.worker.memory.spill": 0.9,
            "distributed.worker.memory.target": 0.85,
            "distributed.worker.memory.pause": 0.95,
            "distributed.worker.memory.terminate": False,
            "distributed.worker.use-file-locking": False,
        }
    )
