"""PyTorch Lightning training script for the weather forecasting model"""
import pytorch_lightning as pl
from graph_weather import GraphWeatherForecaster
import xarray as xr
from torch.utils.data import DataLoader, Dataset
import numpy as np
from graph_weather.models.losses import NormalizedMSELoss
import torchvision.transforms as transforms
import torch


class AnalysisDataset(Dataset):
    def __init__(self, filepaths, coarsen: int = 8):
        super().__init__()
        self.filepaths = filepaths
        self.coarsen = coarsen

    def __len__(self):
        return len(self.filepaths) - 1

    def __getitem__(self, item):
        start = (
            xr.open_zarr(self.filepaths[item], consolidated=True)
            .coarsen(latitude=self.coarsen, boundary="pad")
            .mean()
            .coarsen(longitude=self.coarsen)
            .mean()
        )
        end = (
            xr.open_zarr(self.filepaths[item + 1], consolidated=True)
            .coarsen(latitude=self.coarsen, boundary="pad")
            .mean()
            .coarsen(longitude=self.coarsen)
            .mean()
        )

        # Stack the data into a large data cube
        input_data = np.stack(
            [
                start[f"{var}"].values
                for var in start.data_vars
                if not np.isnan(start[f"{var}"].values).any()
            ],
            axis=-1,
        )
        mean = np.mean(input_data, axis=(0, 1))
        std = np.std(input_data, axis=(0, 1))
        output_data = np.stack(
            [
                end[f"{var}"].values
                for var in end.data_vars
                if not np.isnan(end[f"{var}"].values).any()
            ],
            axis=-1,
        )
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        # Normalize now
        return (
            transform(input_data).transpose(0, 1).reshape(-1, input_data.shape[-1]),
            transform(output_data).transpose(0, 1).reshape(-1, input_data.shape[-1]),
        )


class LitGraphForecaster(pl.LightningModule):
    def __init__(
        self, lat_lons: list, feature_dim: int = 289, hidden_dim: int = 256, lr: float = 3e-4
    ):
        super().__init__()
        self.model = GraphWeatherForecaster(
            lat_lons,
            feature_dim=feature_dim,
            hidden_dim_decoder=hidden_dim,
            hidden_dim_processor_node=hidden_dim,
            hidden_layers_processor_edge=hidden_dim,
            hidden_dim_processor_edge=hidden_dim,
        )
        self.criterion = NormalizedMSELoss(
            lat_lons=lat_lons, feature_variance=np.ones((feature_dim,))
        )
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
