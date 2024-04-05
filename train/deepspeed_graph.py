"""Module for training the Graph Weather forecaster model using PyTorch Lightning."""

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from graph_weather import GraphWeatherForecaster

lat_lons = []
for lat in range(-90, 90, 1):
    for lon in range(0, 360, 1):
        lat_lons.append((lat, lon))


class LitModel(pl.LightningModule):
    """
    LightningModule for the weather forecasting model.

    Args:
        lat_lons: List of latitude and longitude coordinates.
        feature_dim: Dimension of the input features.
        aux_dim : Dimension of the auxiliary features.

    Methods:
        __init__: Initialize the LitModel object.
    """

    def __init__(self, lat_lons, feature_dim, aux_dim):
        """
        Initialize the LitModel object.

        Args:
            lat_lons: List of latitude and longitude coordinates.
            feature_dim : Dimension of the input features.
            aux_dim : Dimension of the auxiliary features.
        """
        super().__init__()
        self.model = GraphWeatherForecaster(
            lat_lons=lat_lons, feature_dim=feature_dim, aux_dim=aux_dim
        )

    def training_step(self, batch):
        """
        Performs a  training step.

        Args:
            batch: A batch of training data.

        Returns:
            The computed loss.
        """
        x, y = batch
        x = x.half()
        y = y.half()
        out = self.forward(x)
        criterion = torch.nn.MSELoss()
        loss = criterion(out, y)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer used during training.

        Returns:
            The optimizer.
        """
        return torch.optim.AdamW(self.parameters())

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the model.
        """
        return self.model(x)


class FakeDataset(Dataset):
    """
    Dataset class for generating fake data.

    Methods:
        __init__: Initialize the FakeDataset object.
        __len__: Return the length of the dataset.
        __getitem__: Get an item from the dataset.
    """

    def __init__(self):
        """
        Initialize the FakeDataset object.
        """
        super(FakeDataset, self).__init__()

    def __len__(self):
        return 64000

    def __getitem__(self, item):
        return torch.randn((64800, 605 + 32)), torch.randn((64800, 605))


model = LitModel(lat_lons=lat_lons, feature_dim=605, aux_dim=32)
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    strategy="deepspeed_stage_3_offload",
    precision=16,
    max_epochs=10,
    limit_train_batches=2000,
)
dataset = FakeDataset()
train_dataloader = DataLoader(
    dataset, batch_size=1, num_workers=1, pin_memory=True, prefetch_factor=1
)
trainer.fit(model=model, train_dataloaders=train_dataloader)
