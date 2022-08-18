from typing import Tuple, List

import numpy as np
import torch
import pytorch_lightning as pl

from graph_weather.models.forecast import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss


class LitGraphForecaster(pl.LightningModule):
    def __init__(
        self,
        lat_lons: List,
        feature_dim: int,
        aux_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 3,
        lr: float = 1e-3,
        rollout: int = 1,
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
        self.feature_dim = feature_dim
        self.lr = lr
        self.rollout = rollout
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gnn(x)

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        assert len(batch) == (self.rollout + 1), "Rollout window doesn't match len(batch)!"
        train_loss = torch.zeros(1, dtype=batch[0].dtype, device=self.device, requires_grad=False)
        # start rollout
        x = batch[0]
        for rstep in range(self.rollout):
            y_hat = self(x)  # prediction at rollout step rstep
            y = batch[rstep + 1]  # target
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            train_loss += self.loss(y_hat, y[..., : self.feature_dim])
            # autoregressive predictions - we re-init the "variable" part of x
            x[..., : self.feature_dim] = y_hat
        self.log("train_wmse", train_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        val_loss = self._shared_eval_step(batch, batch_idx)
        self.log("val_wmse", val_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        test_loss = self._shared_eval_step(batch, batch_idx)
        self.log("test_wmse", test_loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return test_loss

    def predict_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        preds: List[torch.Tensor] = []
        with torch.no_grad():
            # start rollout
            x = batch[0]
            for _ in range(self.rollout):
                y_hat = self(x)
                x[..., : self.feature_dim] = y_hat
                preds.append(y_hat)
        return torch.stack(preds, dim=-1)  # stack along new last dimension

    def _shared_eval_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        del batch_idx
        assert len(batch) == (self.rollout + 1), "Rollout window doesn't match len(batch)!"
        loss = torch.zeros(1, dtype=batch[0].dtype, device=self.device, requires_grad=False)
        with torch.no_grad():
            # start rollout
            x = batch[0]
            for rstep in range(self.rollout):
                y_hat = self(x)
                y = batch[rstep + 1]
                loss += self.loss(y_hat, y[..., : self.feature_dim])
                x[..., : self.feature_dim] = y_hat
        return loss

    def configure_optimizers(self):
        # TODO: add a learn rate scheduler?
        return torch.optim.Adam(self.parameters(), betas=(0.0, 0.999), lr=self.lr)
        # return torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
