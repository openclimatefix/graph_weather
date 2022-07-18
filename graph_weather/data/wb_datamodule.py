from typing import Callable, List, Tuple, Optional
import os
import glob

import dask.array as da
from dask.distributed import Client
from einops import rearrange
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from graph_weather.data.wb_dataset import WeatherBenchDataset
from graph_weather.data.wb_constants import WeatherBenchConstantFields
from graph_weather.utils.config import YAMLConfig


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


class WeatherBenchDataModule(pl.LightningDataModule):
    def __init__(self, config: YAMLConfig, dask_client: Optional[Client] = None) -> None:
        super().__init__()
        self.batch_size = config["model:dataloader:batch-size"]
        self.num_workers = config["model:dataloader:num-workers"]

        self.ds_train = WeatherBenchDataset(
            fnames=glob.glob(
                os.path.join(config[f"input:variables:training:basedir"], config[f"input:variables:training:filename-template"])
            ),
            var_names=config["input:variables:names"],
            lead_time=config["model:lead-time"],
            batch_chunk_size=config["model:dataloader:batch-chunk-size"],
            var_means=None,
            var_std=None,
            dask_client=dask_client,
            persist_in_memory=config["model:dask:persist-data"],
        )

        self.const_data = WeatherBenchConstantFields(
            const_fname=config["input:constants:filename"],
            const_names=config["input:constants:names"],
            batch_chunk_size=config["model:dataloader:batch-chunk-size"],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            # we're putting together one full batch from this many batch-chunks
            # this means the "real" batch size == config["model:dataloader:batch-size"] * config["model::dataloader:batch-chunk-size"]
            batch_size=self.batch_size,
            # num_workers > 0 will lead to a deadlock, even with zarr!
            # TODO: figure out if we can get around this limitation (w/o splitting the data into many files)
            # https://discuss.pytorch.org/t/problems-using-dataloader-for-dask-xarray-netcdf-data/108270
            num_workers=self.num_workers,
            # uses pinned memory to speed up CPU-to-GPU data transfers
            # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
            pin_memory=True,
            # enable shuffling (off by default!)
            shuffle=True,
            # custom collator (see above)
            collate_fn=_custom_collator_wrapper(self.const_data.constants),
        )
