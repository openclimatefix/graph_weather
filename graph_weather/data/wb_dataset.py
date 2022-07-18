# Dataloader for WeatherBench (netCDF)
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

import dask.array as da
from dask.distributed import Client

from torch.utils.data import Dataset

__DEBUG__ = False
N_TIMES = 200


class WeatherBenchDataset(Dataset):
    """
    Map-style dataset that returns individual batches rather than single samples.
    A mini-batch is assembled from several "chunks" (not related to the Dask chunks).
    The number of chunks in a batch should be set through the DataLoader.
    This allows us to randomly shuffle chunks inside a batch, hence individual batches will be different between epochs.
    """

    # chunk size for dask arrays along the time dimension
    # larger chunk size will lead to weird errors, so keep this small
    __dask_time_chunk = 10

    def __init__(
        self,
        fnames: List[str],
        var_names: List[str],
        lead_time: int = 6,
        batch_chunk_size: int = 4,
        var_means: Optional[xr.Dataset] = None,
        var_std: Optional[xr.Dataset] = None,
        dask_client: Optional[Client] = None,
        persist_in_memory: bool = False,
    ) -> None:
        """
        Args:
            fnames: list of data files to open
            var_names: list of variable names to retain
            lead_time: lead time, in hours (must be a multiple of 6)
            batch_chunk_size: size of a batch chunk
            var_means: pre-computed means
            var_std: pre-computed standard deviations
            dask_client: dask Client, used for parallel data processing, batching, etc.
            persist_in_memory: if True and (dask_client is not None), persist dataset in RAM, distributed across all Dask workers
                               if True and (dask_client == None), we simply load the data in the memory of the current process
        """
        super().__init__()
        self.vars = var_names
        self.bs = batch_chunk_size
        # Dask client (parallel data processing)
        self.dask_client = dask_client

        # assumes hourly data, otherwise the length is incorrect
        if (lead_time <= 0) or (lead_time % 6 != 0):
            raise RuntimeError(f"Lead time = {lead_time}, but it must be a (positive) multiple of 6!")
        self.lead_time = lead_time

        # open data files and retain only the variables we need
        self.ds = xr.open_mfdataset(fnames, parallel=True, chunks={"time": self.__dask_time_chunk})
        self.ds = self.ds[var_names]

        self.nvar = len(self.vars)
        self.nlev = len(self.ds.level)
        self.nlat, self.nlon = len(self.ds.latitude), len(self.ds.longitude)

        if __DEBUG__:
            # useful when debugging: retain only the first N_TIMES samples
            self.ds = self.ds.isel(time=slice(None, N_TIMES))

        self.length = int(np.ceil((len(self.ds.time) - lead_time) / self.bs))
        print(f"Dataset length: {self.length}")

        # this will trigger the computation of the mean and std (if needed)
        if persist_in_memory:
            self.ds = self.__persist(self.ds)

        # normalization (mu, std)
        self.mean = self.ds.mean().compute() if var_means is None else var_means
        self.std = self.ds.std("time").mean(("level", "latitude", "longitude")).compute() if var_std is None else var_std

    def _transform(self, data: xr.Dataset) -> xr.Dataset:
        return (data - self.mean) / self.std

    def __len__(self):
        """Returns the length of the dataset"""
        return self.length

    def __persist(self, ds: xr.Dataset) -> xr.Dataset:
        return self.dask_client.persist(ds) if self.dask_client is not None else ds.load()

    def __getitem__(self, i: int) -> Tuple[np.ndarray, ...]:
        return self.__get_mini_batch_chunk(i)

    def __get_mini_batch_chunk(self, i) -> Tuple[np.ndarray, ...]:
        """Returns (part of) a mini-batch"""
        start, end = i * self.bs, (i + 1) * self.bs
        Xv_ = self._transform(self.ds.isel(time=slice(start, end)))

        start, end = i * self.bs + self.lead_time, (i + 1) * self.bs + self.lead_time
        Yv_ = self._transform(self.ds.isel(time=slice(start, end)))

        # shape: (bs, nvar, nlev, lat, lon)
        X = da.stack([Xv_[var] for var in self.vars], axis=1)
        Y = da.stack([Yv_[var] for var in self.vars], axis=1)

        return X, Y
