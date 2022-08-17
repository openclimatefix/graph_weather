from typing import Tuple, List

import numpy as np
import torch
import pytorch_lightning as pl

from graph_weather.models.forecast import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss


class LitGraphForecaster(pl.LightningModule):
    def __init__(
        self, lat_lons: List, feature_dim: int, aux_dim: int, hidden_dim: int = 64, num_blocks: int = 3, lr: float = 3e-4
    ) -> None:
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
        self.loss = NormalizedMSELoss(feature_variance=np.ones((feature_dim,)), lat_lons=lat_lons)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gnn(x)

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_wmse", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        x, y = batch
        with torch.no_grad():
            y_hat = self(x)
            val_loss = self.loss(y_hat, y)
            self.log("val_wmse", val_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        # TODO: add a learn rate scheduler?
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
