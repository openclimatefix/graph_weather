from typing import Callable, List, Tuple, Optional
import os
import glob
from functools import partial

import xarray as xr
import dask.array as da
from dask.distributed import Client
from einops import rearrange
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from graph_weather.data.wb_dataset import WeatherBenchDataset, worker_init_func
from graph_weather.data.wb_constants import WeatherBenchConstantFields
from graph_weather.utils.config import YAMLConfig
from graph_weather.utils.dask_utils import init_dask_client
from graph_weather.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _custom_collator_wrapper(const_data: np.ndarray) -> Callable:
    def custom_collator(batch_data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom collation function. It collates several batch chunks into a "full" batch.
        Args:
            data: batch data, [(X_chunk0, Y_chunk0), (X_chunk1, Y_chunk1), ...]
                  with X_chunk0.shape == (batch_chunk_size, nvars, nlevels, lat, lon)
        """
        X, Y = list(zip(*batch_data))
        return torch.as_tensor(
            np.concatenate(
                [
                    # reshape to (bs, (lat*lon), (nvar * nlev))
                    rearrange(da.concatenate([x for x in X], axis=0).compute(), "b v l h w -> b (h w) (v l)"),
                    # reshape to (bs, (lat*lon), nconst)
                    rearrange(np.concatenate([const_data for x in X], axis=0), "b h w c -> b (h w) c"),
                ],
                # concat along last axis (var index)
                axis=-1,
            )
        ), torch.as_tensor(
            rearrange(da.concatenate([y for y in Y], axis=0).compute(), "b v l h w -> b (h w) (v l)"),
        )

    return custom_collator


def get_weatherbench_dataset(fnames: List[str], config: YAMLConfig, scheduler_address: Optional[str] = None) -> xr.Dataset:
    client: Optional[Client] = init_dask_client(scheduler_address, config) if scheduler_address is not None else None
    return xr.open_mfdataset(
        fnames,
        parallel=True,
        chunks={"time": 10},
        lock=False,  # uses Dask if a client is present
    )


class WeatherBenchDataModule(pl.LightningDataModule):
    def __init__(self, config: YAMLConfig, scheduler_address: Optional[str] = None) -> None:
        super().__init__()
        self.batch_size = config["model:dataloader:batch-size"]
        self.num_workers = config["model:dataloader:num-workers"]
        self.config = config

        var_means, var_sds = self._calculate_summary_statistics(scheduler_address)

        self.ds_train = WeatherBenchDataset(
            fnames=glob.glob(
                os.path.join(config["input:variables:training:basedir"], config["input:variables:training:filename-template"])
            ),
            var_names=config["input:variables:names"],
            read_wb_data_func=partial(get_weatherbench_dataset, config=config, scheduler_address=scheduler_address),
            var_mean=var_means,
            var_sd=var_sds,
            lead_time=config["model:lead-time"],
            batch_chunk_size=config["model:dataloader:batch-chunk-size"],
        )

        self.ds_valid = WeatherBenchDataset(
            fnames=glob.glob(
                os.path.join(config["input:variables:validation:basedir"], config["input:variables:validation:filename-template"])
            ),
            var_names=config["input:variables:names"],
            read_wb_data_func=partial(get_weatherbench_dataset, config=config, scheduler_address=scheduler_address),
            var_mean=self.ds_train.mean,
            var_sd=self.ds_train.sd,
            lead_time=config["model:lead-time"],
            batch_chunk_size=config["model:dataloader:batch-chunk-size"],
        )

        self.const_data = WeatherBenchConstantFields(
            const_fname=config["input:constants:filename"],
            const_names=config["input:constants:names"],
            batch_chunk_size=config["model:dataloader:batch-chunk-size"],
        )

    def _calculate_summary_statistics(self, dask_cluster_address: Optional[str] = None) -> Tuple[xr.Dataset, xr.Dataset]:
        if dask_cluster_address is not None:
            client = Client(dask_cluster_address)

        with xr.open_mfdataset(
            glob.glob(
                os.path.join(
                    self.config["input:variables:training:basedir"], self.config["input:variables:training:filename-template"]
                )
            ),
            chunks={"time": 10},
            parallel=True,  # uses Dask if a client is present
        ) as ds_wb:
            ds_wb = ds_wb[self.config["input:variables:names"]]
            var_means = ds_wb.mean().compute()
            var_sds = ds_wb.std("time").mean(("level", "latitude", "longitude")).compute()

        return var_means, var_sds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            # we're putting together one full batch from this many batch-chunks
            # this means the "real" batch size == config["model:dataloader:batch-size"] * config["model::dataloader:batch-chunk-size"]
            batch_size=self.batch_size,
            # number of worker processes
            num_workers=self.num_workers,
            # uses pinned memory to speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=False,
            # custom collator (see above)
            collate_fn=_custom_collator_wrapper(self.const_data.constants),
            # worker initializer
            worker_init_fn=worker_init_func,
            # prefetch batches (default prefetch_factor == 2)
            prefetch_factor=4,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_custom_collator_wrapper(self.const_data.constants),
            worker_init_fn=worker_init_func,
            prefetch_factor=4,
        )
