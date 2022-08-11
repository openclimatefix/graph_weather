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
        self.vars = var_names
        self.nvar = len(self.vars)
        self.bs = batch_chunk_size
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
        self.effective_ds_len = int(np.ceil((len(self.ds.time) - self.lead_time) / self.bs))
        self.n_samples_per_epoch_per_worker = self.effective_ds_len // n_workers

        # each worker must have a different seed for its random number generator,
        # otherwise all the workers will output exactly the same data
        self.rng = np.random.default_rng(seed=torch.initial_seed())

    def _transform(self, data: xr.Dataset) -> xr.Dataset:
        return (data - self.mean) / self.sd

    def __iter__(self):
        LOGGER.debug(f"n_samples_per_epoch_per_worker: {self.n_samples_per_epoch_per_worker}")
        for _ in range(self.n_samples_per_epoch_per_worker):
            i = self.rng.integers(low=0, high=self.effective_ds_len, dtype=np.uint32)

            start, end = i * self.bs, (i + 1) * self.bs
            Xv_ = self._transform(self.ds.isel(time=slice(start, end)))

            start, end = i * self.bs + self.lead_time, (i + 1) * self.bs + self.lead_time
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


# class WeatherBenchDataset(Dataset):
#     """
#     Map-style dataset that returns individual batches rather than single samples.
#     A mini-batch is assembled from several "chunks" (not related to the Dask chunks).
#     The number of chunks in a batch should be set through the DataLoader.
#     This allows us to randomly shuffle chunks inside a batch, hence individual batches will be different between epochs.
#     """

#     # chunk size for dask arrays along the time dimension
#     # larger chunk size will lead to weird errors, so keep this small
#     __dask_time_chunk = 10

#     def __init__(
#         self,
#         fnames: List[str],
#         var_names: List[str],
#         lead_time: int = 6,
#         batch_chunk_size: int = 4,
#         var_means: Optional[xr.Dataset] = None,
#         var_std: Optional[xr.Dataset] = None,
#         dask_client: Optional[Client] = None,
#         persist_in_memory: bool = False,
#     ) -> None:
#         """
#         Args:
#             fnames: list of data files to open
#             var_names: list of variable names to retain
#             lead_time: lead time, in hours (must be a multiple of 6)
#             batch_chunk_size: size of a batch chunk
#             var_means: pre-computed means
#             var_std: pre-computed standard deviations
#             dask_client: dask Client, used for parallel data processing, batching, etc.
#             persist_in_memory: if True and (dask_client is not None), persist dataset in RAM, distributed across all Dask workers
#                                if True and (dask_client == None), we simply load the data in the memory of the current process
#         """
#         super().__init__()
#         self.vars = var_names
#         self.bs = batch_chunk_size
#         # Dask client (parallel data processing)
#         self.dask_client = dask_client

#         # assumes hourly data, otherwise the length is incorrect
#         if (lead_time <= 0) or (lead_time % 6 != 0):
#             raise RuntimeError(f"Lead time = {lead_time}, but it must be a (positive) multiple of 6!")
#         self.lead_time = lead_time

#         # open data files and retain only the variables we need
#         self.ds = xr.open_mfdataset(fnames, parallel=True, chunks={"time": self.__dask_time_chunk})
#         self.ds = self.ds[var_names]

#         self.nvar = len(self.vars)
#         self.nlev = len(self.ds.level)
#         self.nlat, self.nlon = len(self.ds.latitude), len(self.ds.longitude)

#         if __DEBUG__:
#             # useful when debugging: retain only the first N_TIMES samples
#             self.ds = self.ds.isel(time=slice(None, N_TIMES))

#         self.length = int(np.ceil((len(self.ds.time) - lead_time) / self.bs))
#         print(f"Dataset length: {self.length}")

#         # this will trigger the computation of the mean and std (if needed)
#         if persist_in_memory:
#             self.ds = self.__persist(self.ds)

#         # normalization (mu, std)
#         self.mean = self.ds.mean().compute() if var_means is None else var_means
#         self.std = self.ds.std("time").mean(("level", "latitude", "longitude")).compute() if var_std is None else var_std

#     def _transform(self, data: xr.Dataset) -> xr.Dataset:
#         return (data - self.mean) / self.std

#     def __len__(self):
#         """Returns the length of the dataset"""
#         return self.length

#     def __persist(self, ds: xr.Dataset) -> xr.Dataset:
#         return self.dask_client.persist(ds) if self.dask_client is not None else ds.load()

#     def __getitem__(self, i: int) -> Tuple[np.ndarray, ...]:
#         return self.__get_mini_batch_chunk(i)

#     def __get_mini_batch_chunk(self, i) -> Tuple[np.ndarray, ...]:
#         """Returns (part of) a mini-batch"""
#         start, end = i * self.bs, (i + 1) * self.bs
#         Xv_ = self._transform(self.ds.isel(time=slice(start, end)))

#         start, end = i * self.bs + self.lead_time, (i + 1) * self.bs + self.lead_time
#         Yv_ = self._transform(self.ds.isel(time=slice(start, end)))

#         # shape: (bs, nvar, nlev, lat, lon)
#         X = da.stack([Xv_[var] for var in self.vars], axis=1)
#         Y = da.stack([Yv_[var] for var in self.vars], axis=1)

#         return X, Y
