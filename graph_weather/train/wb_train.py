# Train GNN model on the WeatherBench dataset
from typing import Optional
import argparse
import datetime as dt
import os

from dask.distributed import Client
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from graph_weather.utils.dask_utils import init_dask
from graph_weather.utils.config import YAMLConfig
from graph_weather.data.wb_datamodule import WeatherBenchDataModule
from graph_weather.utils.logger import get_logger
from graph_weather.train.wb_trainer import LitGraphForecaster

LOGGER = get_logger(__name__)


def train(config: YAMLConfig) -> None:
    """
    Train entry point.
    Args:
        config: job configuration
    """

    # initialize dask cluster
    client: Optional[Client] = init_dask(config) if config["model:dask:enabled"] else None

    dmod = WeatherBenchDataModule(config, dask_client=client)

    # number of variables (features)
    num_features = dmod.ds_train.nlev * dmod.ds_train.nvar
    LOGGER.debug(f"GNN num_features = {num_features}")
    LOGGER.debug(f"GNN aux_dim = {dmod.const_data.nconst}")

    model = LitGraphForecaster(
        lat_lons=dmod.const_data.latlons,
        feature_dim=num_features,
        aux_dim=dmod.const_data.nconst,
        hidden_dim=config["model:hidden-dim"],
        num_blocks=config["model:num-blocks"],
        lr=config["model:learn-rate"],
    )

    # init logger
    if config["model:wandb:enabled"]:
        # use weights-and-biases
        logger = WandbLogger(
            project="GNN-WB",
            save_dir=config["output:logging:log-dir"],
        )
    else:
        # use tensorboard
        logger = TensorBoardLogger(config["output:logging:log-dir"])

    # fast_dev_run -> runs a single batch
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=[
            EarlyStopping(monitor="mse_train", min_delta=1.0e-2, patience=3, verbose=False, mode="min"),
            ModelCheckpoint(
                dirpath=os.path.join(
                    config["output:basedir"],
                    dt.datetime.now().strftime("%Y%m%d_%H%M"),
                ),
                filename=config[f"output:model:checkpoint-filename"],
                monitor="mse_train",
                verbose=False,
                save_top_k=config["output:model:save-top-k"],
                save_weights_only=True,
                mode="min",
                auto_insert_metric_name=True,
                save_on_train_epoch_end=True,
                every_n_epochs=1,
            ),
        ],
        detect_anomaly=config[f"model:debug:anomaly-detection"],
        devices=config[f"model:num-gpus"],
        precision=config[f"model:precision"],
        max_epochs=config[f"model:max-epochs"],
        logger=logger,
        log_every_n_steps=config["output:logging:log-interval"],
        limit_train_batches=config[f"model:limit-train-batches"],
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#fast-dev-run
        # fast_dev_run=config["output:logging:fast-dev-run"],
    )

    trainer.fit(model, datamodule=dmod)

    LOGGER.debug("---- DONE. ----")


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--config", required=True, help="Model configuration file (YAML)")
    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    args = get_args()
    config = YAMLConfig(args.config)
    train(config)


if __name__ == "__main__":
    train()
