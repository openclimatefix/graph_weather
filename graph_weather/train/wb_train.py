# Train GNN model on the WeatherBench dataset
import argparse

from dask.distributed import Client

from graph_weather.utils.dask_utils import init_dask
from graph_weather.utils.config import YAMLConfig


def train(config: YAMLConfig):
    """Train entry point"""
    client: Client = init_dask(config)


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
