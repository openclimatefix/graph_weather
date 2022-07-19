from typing import Tuple, List

import numpy as np
import torch
import pytorch_lightning as pl

from graph_weather.models.forecast import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss


class LitGraphForecaster(pl.LightningModule):
    def __init__(
        self, lat_lons: List, feature_dim: int = 605, aux_dim: int = 6, hidden_dim: int = 64, num_blocks: int = 3, lr: float = 3e-4
    ):
        super().__init__()
        self.gnn = GraphWeatherForecaster(
            lat_lons,
            feature_dim=feature_dim,
            aux_dim=aux_dim,
            hidden_dim_decoder=hidden_dim,
            hidden_dim_processor_node=hidden_dim,
            hidden_layers_processor_edge=hidden_dim,
            hidden_dim_processor_edge=hidden_dim,
            num_blocks=num_blocks,
        )
        self.criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=np.ones((feature_dim,)))
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        return self.gnn(x)

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        x, y = batch
        # TODO: remove this check
        if torch.isnan(x).any() or torch.isnan(y).any():
            raise Exception("NaNs detected in the input data! Do something!!!")
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("mse_train", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # TODO: add a learn rate scheduler?
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
