# Train GNN model on the WeatherBench dataset
from typing import Optional
import argparse

from dask.distributed import Client
import pytorch_lightning as pl

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
    LOGGER.debug(f"num_features = {num_features}")

    model = LitGraphForecaster(
        lat_lons=dmod.const_data.latlons,
        feature_dim=num_features,
        aux_dim=dmod.const_data.nconst,
        hidden_dim=config["model:hidden-dim"],
        num_blocks=config["model:num-blocks"],
        lr=config["model:learn-rate"],
    )

    # fast_dev_run -> runs a single batch
    trainer = pl.Trainer(accelerator="gpu", max_epochs=2, precision=32, fast_dev_run=True)
    trainer.fit(model, dmod.train_dataloader())

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
