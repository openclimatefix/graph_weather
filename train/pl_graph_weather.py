"""PyTorch Lightning training script for the weather forecasting model"""

import click
import datasets
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pysolar.util import extraterrestrial_irrad
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from graph_weather import GraphWeatherForecaster
from graph_weather.data import const
from graph_weather.models.losses import NormalizedMSELoss

const.FORECAST_MEANS = {var: np.asarray(value) for var, value in const.FORECAST_MEANS.items()}
const.FORECAST_STD = {var: np.asarray(value) for var, value in const.FORECAST_STD.items()}


def worker_init_fn(worker_id):
    """
    Initialize random seed for worker.

    Args:
        worker_id (int): ID of the worker.

    Returns:
        None
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_mean_stds():
    """
    Calculate means and standard deviations for forecast variables.

    Returns:
        means and standard deviations dict for forecast variables
    """
    names = [
        "CLMR",
        "GRLE",
        "VVEL",
        "VGRD",
        "UGRD",
        "O3MR",
        "CAPE",
        "TMP",
        "PLPL",
        "DZDT",
        "CIN",
        "HGT",
        "RH",
        "ICMR",
        "SNMR",
        "SPFH",
        "RWMR",
        "TCDC",
        "ABSV",
    ]
    means = {}
    stds = {}
    # For pressure level values
    for n in names:
        if (
            len(
                sorted(
                    [
                        float(var.split(".", 1)[-1].split("_")[0])
                        for var in const.FORECAST_MEANS
                        if "mb" in var and n in var and "-" not in var
                    ]
                )
            )
            > 0
        ):
            means[n + "_mb"] = []
            stds[n + "_mb"] = []
            for value in sorted(
                [
                    float(var.split(".", 1)[-1].split("_")[0])
                    for var in const.FORECAST_MEANS
                    if "mb" in var and n in var and "-" not in var
                ]
            ):
                # Is floats now, but will be fixed
                if value >= 1:
                    value = int(value)
                var_name = f"{n}.{value}_mb"
                # print(var_name)

                means[n + "_mb"].append(const.FORECAST_MEANS[var_name])
                stds[n + "_mb"].append(const.FORECAST_STD[var_name])
            means[n + "_mb"] = np.mean(np.stack(means[n + "_mb"], axis=-1))
            stds[n + "_mb"] = np.mean(np.stack(stds[n + "_mb"], axis=-1))

    # For surface values
    for n in list(
        set(
            [
                var.split(".", 1)[0]
                for var in const.FORECAST_MEANS
                if "surface" in var
                and "level" not in var
                and "2e06" not in var
                and "below" not in var
                and "atmos" not in var
                and "tropo" not in var
                and "iso" not in var
                and "planetary_boundary_layer" not in var
            ]
        )
    ):
        means[n] = const.FORECAST_MEANS[n + ".surface"]
        stds[n] = const.FORECAST_STD[n + ".surface"]

    # For Cloud levels
    for n in list(
        set(
            [
                var.split(".", 1)[0]
                for var in const.FORECAST_MEANS
                if "sigma" not in var
                and "level" not in var
                and "2e06" not in var
                and "below" not in var
                and "atmos" not in var
                and "tropo" not in var
                and "iso" not in var
                and "planetary_boundary_layer" not in var
            ]
        )
    ):
        if "LCDC" in n:  # or "MCDC" in n or "HCDC" in n:
            means[n] = const.FORECAST_MEANS["LCDC.low_cloud_layer"]
            stds[n] = const.FORECAST_STD["LCDC.low_cloud_layer"]
        if "MCDC" in n:  # or "HCDC" in n:
            means[n] = const.FORECAST_MEANS["MCDC.middle_cloud_layer"]
            stds[n] = const.FORECAST_STD["MCDC.middle_cloud_layer"]
        if "HCDC" in n:
            means[n] = const.FORECAST_MEANS["HCDC.high_cloud_layer"]
            stds[n] = const.FORECAST_STD["HCDC.high_cloud_layer"]

    # Now for each of these
    means["max_wind"] = []
    stds["max_wind"] = []
    for n in sorted([var for var in const.FORECAST_MEANS if "max_wind" in var]):
        means["max_wind"].append(const.FORECAST_MEANS[n])
        stds["max_wind"].append(const.FORECAST_STD[n])
    means["max_wind"] = np.stack(means["max_wind"], axis=-1)
    stds["max_wind"] = np.stack(stds["max_wind"], axis=-1)

    for i in [2, 10, 20, 30, 40, 50, 80, 100]:
        means[f"{i}m_above_ground"] = []
        stds[f"{i}m_above_ground"] = []
        for n in sorted([var for var in const.FORECAST_MEANS if f"{i}_m_above_ground" in var]):
            means[f"{i}m_above_ground"].append(const.FORECAST_MEANS[n])
            stds[f"{i}m_above_ground"].append(const.FORECAST_STD[n])
        means[f"{i}m_above_ground"] = np.stack(means[f"{i}m_above_ground"], axis=-1)
        stds[f"{i}m_above_ground"] = np.stack(stds[f"{i}m_above_ground"], axis=-1)
    return means, stds


means, stds = get_mean_stds()


def process_data(data):
    """Process the input data."""
    data.update(
        {
            key: np.expand_dims(np.asarray(value), axis=-1)
            for key, value in data.items()
            if key.replace("current_", "").replace("next_", "") in means.keys()
            and np.asarray(value).ndim == 2
        }
    )  # Add third dimension for ones with 2
    input_data = {
        key.replace("current_", ""): torch.from_numpy(
            (value - means[key.replace("current_", "")]) / stds[key.replace("current_", "")]
        )
        for key, value in data.items()
        if "current" in key and "time" not in key
    }
    output_data = {
        key.replace("next_", ""): torch.from_numpy(
            (value - means[key.replace("next_", "")]) / stds[key.replace("next_", "")]
        )
        for key, value in data.items()
        if "next" in key and "time" not in key
    }
    lat_lons = np.array(
        np.meshgrid(np.asarray(data["latitude"]).flatten(), np.asarray(data["longitude"]).flatten())
    ).T.reshape((-1, 2))
    sin_lat_lons = np.sin(lat_lons * np.pi / 180.0)
    cos_lat_lons = np.cos(lat_lons * np.pi / 180.0)
    date = pd.to_datetime(data["timestamps"][0], utc=True)
    solar_times = [
        np.array(
            [
                extraterrestrial_irrad(
                    when=date.to_pydatetime(), latitude_deg=lat, longitude_deg=lon
                )
                for lat, lon in lat_lons
            ]
        )
    ]
    for when in pd.date_range(
        date - pd.Timedelta("12 hours"), date + pd.Timedelta("12 hours"), freq="1H"
    ):
        solar_times.append(
            np.array(
                [
                    extraterrestrial_irrad(
                        when=when.to_pydatetime(), latitude_deg=lat, longitude_deg=lon
                    )
                    for lat, lon in lat_lons
                ]
            )
        )
    solar_times = np.array(solar_times)
    # Normalize to between -1 and 1
    solar_times -= const.SOLAR_MEAN
    solar_times /= const.SOLAR_STD
    input_data = torch.concat([value for _, value in input_data.items()], dim=-1)
    output_data = torch.concat([value for _, value in output_data.items()], dim=-1)
    input_data = input_data.transpose(0, 1).reshape(-1, input_data.shape[-1])
    output_data = output_data.transpose(0, 1).reshape(-1, input_data.shape[-1])
    day_of_year = pd.to_datetime(data["timestamps"][0], utc=True).dayofyear / 366.0
    sin_of_year = np.ones_like(lat_lons)[:, 0] * np.sin(day_of_year)
    cos_of_year = np.ones_like(lat_lons)[:, 0] * np.cos(day_of_year)
    to_concat = [
        input_data,
        torch.permute(torch.from_numpy(solar_times), (1, 0)),
        torch.from_numpy(sin_lat_lons),
        torch.from_numpy(cos_lat_lons),
        torch.from_numpy(np.expand_dims(sin_of_year, axis=-1)),
        torch.from_numpy(np.expand_dims(cos_of_year, axis=-1)),
    ]  # , landsea_fixed]
    input_data = torch.concat(to_concat, dim=-1)
    new_data = {
        "input": input_data.float().numpy(),
        "output": output_data.float().numpy(),
        "has_nans": not np.isnan(input_data.float().numpy()).any()
        and not np.isnan(output_data.float().numpy()).any(),
    }
    return new_data


class GraphDataModule(pl.LightningDataModule):
    """
    LightningDataModule for loading graph data.

    Attributes:
        batch_size : Batch size for the dataloader.
        dataset : Dataset containing the loaded data.

    Methods:
        __init__: Initialize the GraphDataModule object by loading and processing data.
        train_dataloader: Create training dataloader.
    """

    def __init__(self, deg: str = "2.0", batch_size: int = 1):
        """
        Initialize the GraphDataModule object by loading and processing data.

        Args:
            deg : Resolution of the dataset.
            batch_size : Batch size for the dataloader.
        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset = datasets.load_dataset(
            "openclimatefix/gfs-surface-pressure-2deg", split="train+validation", streaming=False
        )
        features = datasets.Features(
            {
                "input": datasets.Array2D(shape=(16380, 637), dtype="float32"),
                "output": datasets.Array2D(shape=(16380, 605), dtype="float32"),
                "has_nans": datasets.Value("bool"),
            }
        )
        self.dataset = (
            self.dataset.map(
                process_data,
                remove_columns=self.dataset.column_names,
                features=features,
                num_proc=16,
                writer_batch_size=2,
            )
            .filter(lambda x: x["has_nans"])
            .with_format("torch")
        )

    def train_dataloader(self):
        """
        Create training dataloader.

        Returns:
            torch.utils.data.DataLoader: Training dataloader.
        """
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=2)


class LitGraphForecaster(pl.LightningModule):
    """
    LightningModule for graph-based weather forecasting.

    Attributes:
        model (GraphWeatherForecaster): Graph weather forecaster model.
        criterion (NormalizedMSELoss): Loss criterion for training.
        lr : Learning rate for optimizer.

    Methods:
        __init__: Initialize the LitGraphForecaster object.
        forward: Forward pass of the model.
        training_step: Training step.
        configure_optimizers: Configure the optimizer for training.
    """

    def __init__(
        self,
        lat_lons: list,
        feature_dim: int = 605,
        aux_dim: int = 32,
        hidden_dim: int = 64,
        num_blocks: int = 3,
        lr: float = 3e-4,
    ):
        """
        Initialize the LitGraphForecaster object with the required args.

        Args:
            lat_lons : List of latitude and longitude values.
            feature_dim : Dimensionality of the input features.
            aux_dim : Dimensionality of auxiliary features.
            hidden_dim : Dimensionality of hidden layers in the model.
            num_blocks : Number of graph convolutional blocks in the model.
            lr (float): Learning rate for optimizer.
        """
        super().__init__()
        self.model = GraphWeatherForecaster(
            lat_lons,
            feature_dim=feature_dim,
            aux_dim=aux_dim,
            hidden_dim_decoder=hidden_dim,
            hidden_dim_processor_node=hidden_dim,
            hidden_dim_processor_edge=hidden_dim,
            num_blocks=num_blocks,
        )
        self.criterion = NormalizedMSELoss(
            lat_lons=lat_lons, feature_variance=np.ones((feature_dim,))
        )
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        """
        Forward pass .

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (dict): Batch of data containing input and output tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss tensor.
        """

        x, y = batch["input"], batch["output"]
        if torch.isnan(x).any() or torch.isnan(y).any():
            return None
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


@click.command()
@click.option(
    "--num-blocks",
    default=5,
    help="Where to save the zarr files",
    type=click.INT,
)
@click.option(
    "--hidden",
    default=32,
    help="Where to save the zarr files",
    type=click.INT,
)
@click.option(
    "--batch",
    default=1,
    help="Where to save the zarr files",
    type=click.INT,
)
@click.option(
    "--gpus",
    default=1,
    help="Where to save the zarr files",
    type=click.INT,
)
def run(num_blocks, hidden, batch, gpus):
    """
    Trainig process.

    Args:
        num_blocks : Number of blocks.
        hidden: Hidden dimension.
        batch : Batch size.
        gpus : Number of GPUs.

    Returns:
        None
    """
    hf_ds = datasets.load_dataset(
        "openclimatefix/gfs-surface-pressure-2deg", split="train", streaming=False
    )
    example_batch = next(iter(hf_ds))
    lat_lons = np.array(
        np.meshgrid(
            np.asarray(example_batch["latitude"]).flatten(),
            np.asarray(example_batch["longitude"]).flatten(),
        )
    ).T.reshape((-1, 2))
    checkpoint_callback = ModelCheckpoint(dirpath="./", save_top_k=2, monitor="loss")
    dset = GraphDataModule(batch_size=batch)
    model = LitGraphForecaster(lat_lons=lat_lons, num_blocks=num_blocks, hidden_dim=hidden)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        max_epochs=100,
        precision=16,
        callbacks=[checkpoint_callback],
    )
    # strategy="deepspeed_stage_2_offload")
    trainer.fit(model, dset)


if __name__ == "__main__":
    run()
