from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import xarray
from einops import rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from graph_weather.models import LoRAModule, MetaModel
from graph_weather.models.losses import NormalizedMSELoss


class LitLoRAFengWuGHR(pl.LightningModule):
    def __init__(
        self,
        lat_lons: list,
        single_step_model_state_dict: dict,
        *,
        time_step: int,
        rank: int,
        channels: int,
        image_size,
        patch_size=4,
        depth=5,
        heads=4,
        mlp_dim=5,
        feature_dim: int = 605,  # TODO where does this come from?
        lr: float = 3e-4,
    ):
        super().__init__()
        assert (
            time_step > 1
        ), "Time step must be greater than 1. Remember that 1 is the simple model time step."
        ssmodel = MetaModel(
            lat_lons,
            image_size=image_size,
            patch_size=patch_size,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
        )
        ssmodel.load_state_dict(single_step_model_state_dict)
        self.models = nn.ModuleList(
            [ssmodel] + [LoRAModule(ssmodel, r=rank) for _ in range(2, time_step + 1)]
        )
        self.criterion = NormalizedMSELoss(
            lat_lons=lat_lons, feature_variance=np.ones((feature_dim,))
        )
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        ys = []
        for t, model in enumerate(self.models):
            x = model(x)
            ys.append(x)
        return torch.stack(ys, dim=1)

    def training_step(self, batch, batch_idx):
        if torch.isnan(batch).any():
            return None
        x, ys = batch[:, 0, ...], batch[:, 1:, ...]

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, ys)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class Era5Dataset(Dataset):
    def __init__(self, xarr, time_step=1, transform=None):
        assert time_step > 0, "Time step must be greater than 0."
        ds = np.asarray(xarr.to_array())
        ds = torch.from_numpy(ds)
        ds -= ds.min(0, keepdim=True)[0]
        ds /= ds.max(0, keepdim=True)[0]
        ds = rearrange(ds, "C T H W -> T (H W) C")
        self.ds = ds
        self.time_step = time_step

    def __len__(self):
        return len(self.ds) - self.time_step

    def __getitem__(self, index):
        return self.ds[index : index + time_step + 1]


if __name__ == "__main__":
    ckpt_path = Path("./checkpoints")
    ckpt_name = "best.pt"
    patch_size = 4
    grid_step = 20
    time_step = 2
    rank = 4
    variables = [
        "2m_temperature",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]

    ###############################################################

    channels = len(variables)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    reanalysis = xarray.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        storage_options=dict(token="anon"),
    )

    reanalysis = reanalysis.sel(time=slice("2020-01-01", "2021-01-01"))
    reanalysis = reanalysis.isel(
        time=slice(100, 111), longitude=slice(0, 1440, grid_step), latitude=slice(0, 721, grid_step)
    )

    reanalysis = reanalysis[variables]
    print(f"size: {reanalysis.nbytes / (1024**3)} GiB")

    lat_lons = np.array(
        np.meshgrid(
            np.asarray(reanalysis["latitude"]).flatten(),
            np.asarray(reanalysis["longitude"]).flatten(),
        )
    ).T.reshape((-1, 2))

    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path, save_top_k=1, monitor="loss")

    dset = DataLoader(Era5Dataset(reanalysis, time_step=time_step), batch_size=10, num_workers=8)

    single_step_model_state_dict = torch.load(ckpt_path / ckpt_name)

    model = LitLoRAFengWuGHR(
        lat_lons=lat_lons,
        single_step_model_state_dict=single_step_model_state_dict,
        time_step=time_step,
        rank=rank,
        ##########
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
        strategy="ddp_find_unused_parameters_true",
    )

    trainer.fit(model, dset)
