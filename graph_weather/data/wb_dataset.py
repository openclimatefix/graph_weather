# Dataloader for WeatherBench (netCDF)
from typing import List, Optional, Callable

import numpy as np
import xarray as xr

import torch
import dask.array as da
from torch.utils.data import IterableDataset
from graph_weather.utils.logger import get_logger

LOGGER = get_logger(__name__)


class WeatherBenchDataset(IterableDataset):
    """
    Iterable dataset for WeatherBench data.
    Design inspired by https://github.com/openclimatefix/predict_pv_yield/blob/main/notebooks/20.0_simplify_data_loading.ipynb.
    """

    # TODO: refactor this hardcoded value
    nlev = 13

    def __init__(
        self,
        fnames: List[str],
        var_names: List[str],
        read_wb_data_func: Callable[..., xr.Dataset],
        var_mean: xr.Dataset,
        var_sd: xr.Dataset,
        lead_time: int = 6,
        batch_chunk_size: int = 4,
    ) -> None:
        """
        Initialize (part of) the dataset state.
        Args:
            fnames: glob or list of filenames (full paths)
            var_names: variable names
            read_wb_data_func: user function that opens and returns the WB xr.Dataset (use Dask!)
            var_mean, var_std: precomputed means and standard deviations for all data vars; used to normalize data
            lead_time: lead time
            batch_chunk_size: batch chunk size
        """
        self.fnames = fnames
        self.ds: Optional[xr.Dataset] = None

        self.lead_time = lead_time
        assert self.lead_time > 0 and self.lead_time % 6 == 0, "Lead time must be multiple of 6 hours"
        self.lead_step = lead_time // 6

        self.vars = var_names
        self.nvar = len(self.vars)
        self.bcs = batch_chunk_size
        self.read_wb_data = read_wb_data_func
        self.mean = var_mean
        self.sd = var_sd

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0
        self.effective_ds_len = 0
        self.rng = None

    def per_worker_init(self, n_workers: int = 1) -> None:
        """Called by worker_init_func on each copy of WeatherBenchDataset after the worker process has been spawned."""
        self.ds: xr.Dataset = self.read_wb_data(self.fnames)
        self.ds = self.ds[self.vars]

        assert self.nlev == len(self.ds.level), "Incorrect number of pressure levels!"

        self.ds_len = len(self.ds.time) - self.lead_step
        print(f"Dataset length: {self.ds_len} ...")
        self.effective_ds_len = int(np.ceil(self.ds_len / self.bcs))
        print(f"Batch chunk size: {self.bcs} -- effective dataset length: {self.effective_ds_len} ...")
        self.n_chunks_per_worker = self.effective_ds_len // n_workers
        print(f"Each worker gets {self.n_chunks_per_worker} batch chunks per epoch...")

        # each worker must have a different seed for its random number generator,
        # otherwise all the workers will output exactly the same data
        self.rng = np.random.default_rng(seed=torch.initial_seed())

    def _transform(self, data: xr.Dataset) -> xr.Dataset:
        return (data - self.mean) / self.sd

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
            raise RuntimeError
        else:
            worker_id = worker_info.id
            low = worker_id * self.n_chunks_per_worker
            high = min((worker_id + 1) * self.n_chunks_per_worker, self.ds_len)
            # LOGGER.debug(f"Worker ID {worker_id} owns range {low} -- {high} ...")

        chunk_index_range = np.arange(low, high, dtype=np.uint32)
        shuffled_chunk_indices = self.rng.choice(chunk_index_range, size=self.n_chunks_per_worker, replace=False)

        for i in shuffled_chunk_indices:
            # LOGGER.debug(f"Worker ID {worker_id} sampled index {i} ...")

            start, end = i * self.bcs, (i + 1) * self.bcs
            Xv_ = self._transform(self.ds.isel(time=slice(start, end)))

            start, end = i * self.bcs + self.lead_step, (i + 1) * self.bcs + self.lead_step
            Yv_ = self._transform(self.ds.isel(time=slice(start, end)))

            # shape: (bs, nvar, nlev, lat, lon)
            X = da.stack([Xv_[var] for var in self.vars], axis=1)
            Y = da.stack([Yv_[var] for var in self.vars], axis=1)

            yield (X, Y)


def worker_init_func(worker_id: int) -> None:
    """Configures each dataset worker process by calling WeatherBenchDataset.per_worker_init()."""
    worker_info = torch.utils.data.get_worker_info()  # information specific to each worker process
    if worker_info is None:
        LOGGER.error("worker_info is None! Set num_workers > 0 in your dataloader!")
        raise RuntimeError
    else:
        dataset_obj = worker_info.dataset  # the copy of the dataset held by this worker process.
        dataset_obj.per_worker_init(n_workers=worker_info.num_workers)
