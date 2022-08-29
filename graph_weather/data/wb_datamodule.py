from typing import Callable, List, Tuple, Union
import os
import glob
from functools import partial

import xarray as xr
import dask.array as da
from dask.distributed import Client, LocalCluster
from einops import rearrange
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from graph_weather.data.wb_dataset import WeatherBenchDataset, worker_init_func
from graph_weather.data.wb_constants import WeatherBenchConstantFields
from graph_weather.utils.config import YAMLConfig
from graph_weather.utils.logger import get_logger
import graph_weather.utils.constants as constants

LOGGER = get_logger(__name__)

BatchChunkType = Union[da.Array, List[List[int]]]


class WeatherBenchDataBatch:
    """Custom batch type for WeatherBench data."""

    def __init__(self, batch_data: Tuple[BatchChunkType, ...], const_data: np.ndarray) -> None:
        """Construct a batch object from the variable and constant data tensors."""
        zipped_batch = list(zip(*batch_data))

        batch: List[torch.Tensor] = []
        for X in zipped_batch[:-1]:
            X = torch.as_tensor(
                np.concatenate(
                    [
                        # reshape to (bs, (lat*lon), (nvar * nlev))
                        rearrange(da.concatenate([x for x in X], axis=0).compute(), "b v l h w -> b (h w) (v l)"),
                        # reshape to (bs, (lat*lon), nconst)
                        rearrange(np.concatenate([const_data for x in X], axis=0), "b h w c -> b (h w) c"),
                    ],
                    # concat along last axis (var index)
                    # final shape: (bs, (lat*lon), nvar * nlev + nconst) -> this is what the GNN expects
                    axis=-1,
                )
            )
            batch.append(X)

        self.X: Tuple[torch.Tensor] = tuple(batch)
        self.idx: torch.Tensor = torch.as_tensor(rearrange(np.array(zipped_batch[-1], dtype=np.int32), "bs r bcs -> r (bs bcs)"))

    def pin_memory(self):
        """Custom memory pinning. See https://pytorch.org/docs/stable/data.html#memory-pinning."""
        self.X = tuple(t.pin_memory() for t in self.X)
        self.idx = self.idx.pin_memory()
        return self


def _custom_collator_wrapper(const_data: np.ndarray) -> Callable:
    # TODO: add type annotation
    def custom_collator(batch_data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom collation function. It collates several batch chunks into a "full" batch.
        Args:
            data: batch-chunks of WeatherBench variables, [(X1_chunk0, X2_chunk0, ...), (X1_chunk1, X2_chunk1, ...), ...]
                with Xi_chunk0.shape == (batch_chunk_size, nvars, nlevels, lat, lon)
                and len(X1_chunk0, X2_chunk0, ...) == length of the rollout window + 1
            const_data: constant (time-independent) data
        """
        return WeatherBenchDataBatch(batch_data=batch_data, const_data=const_data)

    return custom_collator


def get_weatherbench_dataset(
    fnames: List[str],
    client: Client,
    config: YAMLConfig,
) -> xr.Dataset:
    LOGGER.debug("Created Dask client %s attached to %s ...", client, client.scheduler_info)
    kwargs = dict(consolidated=True) if config["input:format"] == "zarr" else {}
    return xr.open_mfdataset(
        fnames,
        parallel=(client is not None),  # uses Dask if a client is present
        chunks={"time": constants._DS_TIME_CHUNK},
        engine=config["input:format"],
        **kwargs,
    )


class WeatherBenchTrainingDataModule(pl.LightningDataModule):
    def __init__(self, config: YAMLConfig) -> None:
        super().__init__()
        self.batch_size = config["model:dataloader:training:batch-size"]
        self.num_workers_train = config["model:dataloader:num-workers:training"]
        self.num_workers_val = config["model:dataloader:num-workers:validation"]
        self.config = config

        if config["input:variables:training:summary-stats:precomputed"]:
            var_means, var_sds = self._load_summary_statistics()
        else:
            # use Dask to compute summary statistics for the training data on the fly (can take some time)
            var_means, var_sds = self._calculate_summary_statistics()

        self.ds_train = WeatherBenchDataset(
            fnames=glob.glob(
                os.path.join(config["input:variables:training:basedir"], config["input:variables:training:filename-template"])
            ),
            var_names=config["input:variables:names"],
            read_wb_data_func=partial(get_weatherbench_dataset, config=config),
            var_mean=var_means,
            var_sd=var_sds,
            plevs=config["input:variables:levels"],
            lead_time=config["model:lead-time"],
            batch_chunk_size=config["model:dataloader:training:batch-chunk-size"],
            rollout=config["model:rollout"],
        )

        self.ds_valid = WeatherBenchDataset(
            fnames=glob.glob(
                os.path.join(config["input:variables:validation:basedir"], config["input:variables:validation:filename-template"])
            ),
            var_names=config["input:variables:names"],
            read_wb_data_func=partial(get_weatherbench_dataset, config=config),
            var_mean=var_means,
            var_sd=var_sds,
            plevs=config["input:variables:levels"],
            lead_time=config["model:lead-time"],
            batch_chunk_size=config["model:dataloader:training:batch-chunk-size"],
            rollout=config["model:rollout"],
        )

        self.const_data = WeatherBenchConstantFields(
            const_fname=config["input:constants:filename"],
            const_names=config["input:constants:names"],
            batch_chunk_size=config["model:dataloader:training:batch-chunk-size"],
        )

    def _calculate_summary_statistics(self) -> Tuple[xr.Dataset, xr.Dataset]:
        with LocalCluster(
            n_workers=self.config["model:dask:num-workers"],
            threads_per_worker=self.config["model:dask:num-threads-per-worker"],
            processes=True,
        ) as cluster, Client(cluster) as client:
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

    def _load_summary_statistics(self) -> Tuple[xr.Dataset, xr.Dataset]:
        # load pre-computed means and standard deviations
        var_means = xr.load_dataset(
            os.path.join(
                self.config["input:variables:training:basedir"], self.config["input:variables:training:summary-stats:means"]
            )
        )
        var_sds = xr.load_dataset(
            os.path.join(
                self.config["input:variables:training:basedir"], self.config["input:variables:training:summary-stats:std-devs"]
            )
        )
        return var_means, var_sds

    def _get_dataloader(self, data: xr.Dataset, num_workers: int) -> DataLoader:
        return DataLoader(
            data,
            # we're putting together one full batch from this many batch-chunks
            # this means the "real" batch size == config["model:dataloader:batch-size"] * config["model:dataloader:batch-chunk-size"]
            batch_size=self.batch_size,
            # number of worker processes
            num_workers=num_workers,
            # use of pinned memory can speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=True,
            # custom collator (see above)
            collate_fn=_custom_collator_wrapper(self.const_data.constants),
            # worker initializer
            worker_init_fn=partial(
                worker_init_func,
                dask_temp_dir=self.config["model:dask:temp-dir"],
                num_dask_workers=self.config["model:dask:num-workers"],
                num_dask_threads_per_worker=self.config["model:dask:num-threads-per-worker"],
            ),
            # prefetch batches (default prefetch_factor == 2)
            prefetch_factor=constants._DL_PREFETCH_FACTOR,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_train, self.num_workers_train)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_valid, self.num_workers_val)

    def transfer_batch_to_device(self, batch: WeatherBenchDataBatch, device: torch.device, dataloader_idx: int = 0) -> None:
        del dataloader_idx  # not used
        batch.X = tuple(x.to(device) for x in batch.X)
        batch.idx = batch.idx.to(device)
        return batch


class WeatherBenchTestDataModule(pl.LightningDataModule):
    def __init__(self, config: YAMLConfig) -> None:
        super().__init__()
        self.batch_size = config["model:dataloader:inference:batch-size"]
        self.num_workers = config["model:dataloader:num-workers:inference"]
        self.config = config

        # summary stats must've been precomputed!
        var_means, var_sds = self._load_summary_statistics()

        self.ds_test = WeatherBenchDataset(
            fnames=glob.glob(
                os.path.join(config["input:variables:test:basedir"], config["input:variables:test:filename-template"])
            ),
            var_names=config["input:variables:names"],
            read_wb_data_func=partial(get_weatherbench_dataset, config=config),
            var_mean=var_means,
            var_sd=var_sds,
            plevs=config["input:variables:levels"],
            lead_time=config["model:lead-time"],
            batch_chunk_size=config["model:dataloader:inference:batch-chunk-size"],
            rollout=config["model:rollout"],
        )

        self.ds_predict = WeatherBenchDataset(
            fnames=[config["input:variables:prediction:filename"]],  # single file
            var_names=config["input:variables:names"],
            read_wb_data_func=partial(get_weatherbench_dataset, config=config),
            var_mean=var_means,
            var_sd=var_sds,
            plevs=config["input:variables:levels"],
            lead_time=config["model:lead-time"],
            batch_chunk_size=config["model:dataloader:inference:batch-chunk-size"],
            rollout=config["model:rollout"],
        )

        self.const_data = WeatherBenchConstantFields(
            const_fname=config["input:constants:filename"],
            const_names=config["input:constants:names"],
            batch_chunk_size=config["model:dataloader:inference:batch-chunk-size"],
        )

    def _load_summary_statistics(self) -> Tuple[xr.Dataset, xr.Dataset]:
        # load pre-computed means and standard deviations (calculated from the training data)
        var_means = xr.load_dataset(
            os.path.join(
                self.config["input:variables:training:basedir"], self.config["input:variables:training:summary-stats:means"]
            )
        )
        var_sds = xr.load_dataset(
            os.path.join(
                self.config["input:variables:training:basedir"], self.config["input:variables:training:summary-stats:std-devs"]
            )
        )
        return var_means, var_sds

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_test)

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_predict)

    def _get_dataloader(self, data: xr.Dataset) -> DataLoader:
        return DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_custom_collator_wrapper(self.const_data.constants),
            worker_init_fn=partial(
                worker_init_func,
                dask_temp_dir=self.config["model:dask:temp-dir"],
                num_dask_workers=self.config["model:dask:num-workers"],
                num_dask_threads_per_worker=self.config["model:dask:num-threads-per-worker"],
            ),
            prefetch_factor=constants._DL_PREFETCH_FACTOR,
            persistent_workers=True,
        )

    def transfer_batch_to_device(self, batch: WeatherBenchDataBatch, device: torch.device, dataloader_idx: int = 0) -> None:
        del dataloader_idx  # not used
        batch.X = tuple(x.to(device) for x in batch.X)
        batch.idx = batch.idx.to(device)
        return batch
