from typing import Optional, List
import argparse
import datetime as dt
import os

from dask.distributed import LocalCluster
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import xarray as xr

from graph_weather.utils.dask_utils import init_dask_cluster
from graph_weather.utils.config import YAMLConfig
from graph_weather.data.wb_datamodule import WeatherBenchTestDataModule
from graph_weather.utils.logger import get_logger
from graph_weather.train.wb_trainer import LitGraphForecaster

LOGGER = get_logger(__name__)


def store_predictions(preds: torch.Tensor, config: YAMLConfig) -> None:
    raise NotImplementedError


def backtransform_predictions(preds: torch.Tensor, means: xr.Dataset, sds: xr.Dataset, config: YAMLConfig) -> torch.Tensor:
    raise NotImplementedError


def predict(config: YAMLConfig, checkpoint_filename: str) -> None:
    """
    Predict entry point.
    Args:
        config: job configuration
    """

    # initialize dask cluster
    LOGGER.debug("Initializing Dask cluster ...")
    cluster: Optional[LocalCluster] = init_dask_cluster(config) if config["model:dask:enabled"] else None
    dask_scheduler_address = cluster.scheduler_address if cluster is not None else None
    LOGGER.debug("Dask scheduler address: %s", dask_scheduler_address)

    # create data module (data loaders and data sets)
    dmod = WeatherBenchTestDataModule(config, scheduler_address=dask_scheduler_address)

    # number of variables (features)
    num_features = dmod.ds_test.nlev * dmod.ds_test.nvar
    LOGGER.debug("Number of variables: %d", num_features)
    LOGGER.debug("Number of auxiliary (time-independent) variables: %d", dmod.const_data.nconst)

    model = LitGraphForecaster(
        lat_lons=dmod.const_data.latlons,
        feature_dim=num_features,
        aux_dim=dmod.const_data.nconst,
        hidden_dim=config["model:hidden-dim"],
        num_blocks=config["model:num-blocks"],
        lr=config["model:learn-rate"],
        rollout=config["model:rollout"],
    )

    # TODO: restore model from checkpoint
    checkpoint_filepath = os.path.join(config["output:basedir"], config["output:checkpoints:ckpt-dir"], checkpoint_filename)
    model = LitGraphForecaster.load_from_checkpoint(checkpoint_filepath)

    # init logger
    if config["model:wandb:enabled"]:
        # use weights-and-biases
        logger = WandbLogger(
            project="GNN-WB",
            save_dir=config["output:logging:log-dir"],
        )
    elif config["model:tensorboard:enabled"]:
        # use tensorboard
        logger = TensorBoardLogger(config["output:logging:log-dir"])
    else:
        logger = False

    # fast_dev_run -> runs a single batch
    trainer = pl.Trainer(
        accelerator="gpu",
        detect_anomaly=config["model:debug:anomaly-detection"],
        # devices=config["model:num-gpus"],
        devices=1,  # run on a single GPU... for now
        precision=config["model:precision"],
        logger=logger,
        log_every_n_steps=config["output:logging:log-interval"],
        # run fewer batches per epoch (helpful when debugging)
        limit_test_batches=config["model:limit-test-batches"],
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#fast-dev-run
        # fast_dev_run=config["output:logging:fast-dev-run"],
        max_epochs=-1,
    )

    # run a test loop (calculates test_wmse)
    trainer.test(model, datamodule=dmod)
    # run a predict loop on the same data - same as test in this case
    predictions_ = trainer.predict(model, datamodule=dmod, return_predictions=True)
    LOGGER.debug(predictions_)
    predictions: torch.Tensor = torch.cat(predictions_, dim=0).float().cpu()
    predictions = backtransform_predictions(predictions, config)

    # save data along with observations
    store_predictions(predictions, config)

    LOGGER.debug("---- DONE. ----")


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--config", required=True, help="Model configuration file (YAML)")
    required_args.add_argument(
        "--checkpoint", required=True, help="Name of the model checkpoint file (located under output-basedir/chkpt-dir)."
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for inference."""
    args = get_args()
    config = YAMLConfig(args.config)
    predict(config, args.checkpoint)
