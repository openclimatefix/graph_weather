from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import xarray
from einops import rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from graph_weather.models import MetaModel
from graph_weather.models.losses import NormalizedMSELoss


class LitFengWuGHR(pl.LightningModule):
    """
    LightningModule for graph-based weather forecasting.

    Attributes:
        model (GraphWeatherForecaster): Graph weather forecaster model.
        criterion (NormalizedMSELoss): Loss criterion for training.
        lr : Learning rate for optimizer.

    Methods:
        __init__: Initialize the LitFengWuGHR object.
        forward: Forward pass of the model.
        training_step: Training step.
        configure_optimizers: Configure the optimizer for training.
    """

    def __init__(
        self,
        lat_lons: list,
        *,
        channels: int,
        image_size,
        patch_size=4,
        depth=5,
        heads=4,
        mlp_dim=5,
        feature_dim: int = 605,  # TODO where does this come from?
        lr: float = 3e-4,
    ):
        """
        Initialize the LitFengWuGHR object with the required args.

        Args:
            lat_lons : List of latitude and longitude values.
            feature_dim : Dimensionality of the input features.
            aux_dim : Dimensionality of auxiliary features.
            hidden_dim : Dimensionality of hidden layers in the model.
            num_blocks : Number of graph convolutional blocks in the model.
            lr (float): Learning rate for optimizer.
        """
        super().__init__()
        self.model = MetaModel(
            lat_lons,
            image_size=image_size,
            patch_size=patch_size,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
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
            batch (array): Batch of data containing input and output tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch[:, 0], batch[:, 1]
        if torch.isnan(x).any() or torch.isnan(y).any():
            return None
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class Era5Dataset(Dataset):
    """Era5 dataset."""

    def __init__(self, xarr, transform=None):
        """
        Arguments:
            #TODO
        """
        ds = np.asarray(xarr.to_array())
        ds = torch.from_numpy(ds)
        ds -= ds.min(0, keepdim=True)[0]
        ds /= ds.max(0, keepdim=True)[0]
        ds = rearrange(ds, "C T H W -> T (H W) C")
        self.ds = ds

    def __len__(self):
        return len(self.ds) - 1

    def __getitem__(self, index):
        return self.ds[index : index + 2]


if __name__ == "__main__":

    ckpt_path = Path("./checkpoints")
    patch_size = 4
    grid_step = 20
    variables = [
        "2m_temperature",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]

    channels = len(variables)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    reanalysis = xarray.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        storage_options=dict(token="anon"),
    )

    reanalysis = reanalysis.sel(time=slice("2020-01-01", "2021-01-01"))
    reanalysis = reanalysis.isel(
        time=slice(100, 107), longitude=slice(0, 1440, grid_step), latitude=slice(0, 721, grid_step)
    )

    reanalysis = reanalysis[variables]
    print(f"size: {reanalysis.nbytes / (1024 ** 3)} GiB")

    lat_lons = np.array(
        np.meshgrid(
            np.asarray(reanalysis["latitude"]).flatten(),
            np.asarray(reanalysis["longitude"]).flatten(),
        )
    ).T.reshape((-1, 2))

    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path, save_top_k=1, monitor="loss")

    dset = DataLoader(Era5Dataset(reanalysis), batch_size=10, num_workers=8)
    model = LitFengWuGHR(
        lat_lons=lat_lons,
        channels=channels,
        image_size=(721 // grid_step, 1440 // grid_step),
        patch_size=patch_size,
        depth=5,
        heads=4,
        mlp_dim=5,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=100,
        precision="16-mixed",
        callbacks=[checkpoint_callback],
        log_every_n_steps=3,
    )

    trainer.fit(model, dset)

    torch.save(model.model.state_dict(), ckpt_path / "best.pt")
