"""Fine-tune a pretrained single-step forecaster into a multi-step one with LoRA.

Running this file as a script loads the single-step ``MetaModel`` checkpoint written by
``train/era5.py``, stacks one LoRA-adapted copy per additional lead time on top of it, and
trains the resulting rollout on a slice of the public ARCO-ERA5 zarr store.
"""

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
    """
    LightningModule that rolls a single-step forecaster out over several lead times.

    ``self.models`` holds one entry per lead time: the pretrained single-step
    :class:`~graph_weather.models.fengwu_ghr.layers.MetaModel` first, followed by
    ``time_step - 1`` :class:`~graph_weather.models.fengwu_ghr.layers.LoRAModule` wrappers
    built from it, each attaching rank-``rank`` LoRA layers to the linear layers it can
    reach as direct attributes (see `LoRAModule`). The
    forward pass applies them one after another, feeding each step its predecessor's output.

    Attributes:
        models (nn.ModuleList): The per-step models applied in sequence.
        criterion (NormalizedMSELoss): Loss criterion for training.
        lr : Learning rate for optimizer.
    """

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
        """
        Build the single-step backbone and the LoRA-adapted models stacked on top of it.

        The keyword arguments describing the architecture must match the ones used to train
        ``single_step_model_state_dict``, since that state dict is loaded into a freshly
        constructed :class:`~graph_weather.models.fengwu_ghr.layers.MetaModel`.

        Args:
            lat_lons (list): List of latitude and longitude values, one pair per node of the
                input point cloud.
            single_step_model_state_dict (dict): State dict of an already trained single-step
                model, loaded into the backbone before the LoRA wrappers are built.
            time_step (int): Number of lead times to roll out. Must be greater than 1, since
                1 is the plain single-step model.
            rank (int): Rank of the LoRA layers.
            channels (int): Number of physical variables carried by each node.
            image_size : Height and width of the regular grid the point cloud is interpolated
                onto. A single int is used for both dimensions.
            patch_size : Height and width of the patches the grid is split into before the
                transformer. Both grid dimensions must be divisible by it.
            depth : Number of transformer layers in the backbone.
            heads : Number of attention heads per transformer layer.
            mlp_dim : Hidden dimensionality of the feed-forward block in each transformer
                layer.
            feature_dim (int): Length of the per-feature variance vector handed to
                :class:`NormalizedMSELoss`; a vector of ones of this length is used.
            lr (float): Learning rate for optimizer.
        """
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
        """
        Roll the chain of models out, feeding each step the previous step's prediction.

        Args:
            x (torch.Tensor): Input frame of shape ``[B, N, C]``, where ``N`` is the number of
                lat/lon nodes and ``C`` the number of physical variables.

        Returns:
            torch.Tensor: Prediction of every lead time, stacked along a new axis at dim 1,
            giving shape ``[B, time_step, N, C]``.
        """
        ys = []
        for t, model in enumerate(self.models):
            x = model(x)
            ys.append(x)
        return torch.stack(ys, dim=1)

    def training_step(self, batch, batch_idx):
        """
        Run one training step over a window of consecutive frames.

        Batches holding any NaN are skipped by returning ``None``, which tells Lightning to
        drop the step.

        Args:
            batch (torch.Tensor): Frames of shape ``[B, time_step + 1, N, C]``. The first
                frame is the input and the remaining ones are the targets, one per lead time.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss tensor, or ``None`` when the batch contained NaN values.
        """
        if torch.isnan(batch).any():
            return None
        x, ys = batch[:, 0, ...], batch[:, 1:, ...]

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, ys)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            torch.optim.Optimizer: ``AdamW`` over every parameter of this module, using the
            learning rate given to the constructor.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class Era5Dataset(Dataset):
    """Era5 dataset yielding windows of consecutive time frames."""

    def __init__(self, xarr, time_step=1, transform=None):
        """
        Initialize the dataset by eagerly loading and normalizing the whole slice.

        Args:
            xarr (xarray.Dataset): Reanalysis slice to train on. It is stacked into a dense
                array of shape ``[C, T, H, W]``, min-max scaled using the minimum and maximum
                taken over the leading axis, and then rearranged to ``[T, H * W, C]`` so that
                each time frame is a flat point cloud.
            time_step (int): Number of lead times per sample. Must be greater than 0. It
                shortens the reported length, so that the frames following an index are
                always available as targets.
            transform (callable, optional): Currently unused.
        """
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
