from typing import Callable, List, Tuple, Optional
import os
import glob

import dask.array as da
from dask.distributed import Client
from einops import rearrange
import numpy as np
import torch
from torch.utils.data import DataLoader

from graph_weather.data.dataset import WeatherBenchDataset
from graph_weather.data.constants import ConstantData
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
                    # reshape to (bs, (lat*lon), (nvar * nlev))
                    rearrange(np.concatenate([const_data for x in X], axis=0), "b c h w -> b (h w) c"),
                ],
                axis=1,
            )
        ), torch.as_tensor(
            rearrange(da.concatenate([y for y in Y], axis=0).compute(), "b v l h w -> b (h w) (v l)"),
        )

    return custom_collator


def get_wb_dataloader(config: YAMLConfig, shuffle: bool = False, dask_client: Optional[Client] = None) -> DataLoader:
    """Returns a dataloader object for the WeatherBench dataset."""

    ds_train = WeatherBenchDataset(
        fnames=glob.glob(os.path.join(config["input:basedir"], config["input:filename-template"])),
        var_names=config["input:varnames"],
        lead_time=config["model:lead-time"],
        batch_chunk_size=config["model:batch-chunk-size"],
        var_means=None,
        var_std=None,
        dask_client=dask_client,
        persist_in_memory=config["model:dask:persist-data"],
    )

    const_data = ConstantData().get_constants()

    return DataLoader(
        ds_train,
        # I'm putting together one full batch from 8 batch-chunks
        # The "real" batch size == 8 * 8 == 64
        batch_size=config["model:batch-size"],
        # num_workers > 0 will lead to a deadlock, even with zarr!
        # TODO: figure out if we can get around this limitation (w/o splitting the data into many files)
        # https://discuss.pytorch.org/t/problems-using-dataloader-for-dask-xarray-netcdf-data/108270
        num_workers=config["model:dask:num-workers"],
        # uses pinned memory to speed up CPU-to-GPU data transfers
        # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
        pin_memory=True,
        # enable shuffling (off by default!)
        shuffle=shuffle,
        # custom collator (see above)
        collate_fn=_custom_collator_wrapper(const_data),
    )
