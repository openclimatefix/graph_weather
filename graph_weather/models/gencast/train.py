"""
Training script for GenCast.
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

import lightning as L  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from graph_weather.data.gencast_dataloader import GenCastDataset  # noqa: E402
from graph_weather.models.gencast import Denoiser, Sampler, WeightedMSELoss  # noqa: E402

torch.set_float32_matmul_precision("high")

############################################## SETTINGS ############################################

# training settings
NUM_EPOCHS = 180
NUM_DEVICES = 2
INITIAL_LR = 1e-3
BATCH_SIZE = 16  # per device

# dataloader setting
NUM_WORKERS = 8
PREFETCH_FACTOR = 3
PERSISTENT_WORKERS = True

# model configs
CHECKPOINT_PATH = "checkpoints/epoch=3-step=10776.ckpt"
CFG = {
    "hidden_dims": [512, 512],
    "num_blocks": 16,
    "num_heads": 4,
    "splits": 4,
    "num_hops": 8,
    "sparse": True,
    "use_edges_features": False,
}

# dataset configs
atmospheric_features = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
single_features = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    # "sea_surface_temperature",
    "total_precipitation_12hr",
]
static_features = [
    "geopotential_at_surface",
    "land_sea_mask",
]

OBS_PATH = "dataset.zarr"
# OBS_PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_conservative.zarr"
# OBS_PATH = 'gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
# OBS_PATH = 'gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr'

#################################################################################################


class LitModel(L.LightningModule):
    """Lightning wrapper for Gencast"""

    def __init__(
        self,
        learning_rate,
        cosine_t_max,
        pressure_levels,
        grid_lon,
        grid_lat,
        input_features_dim,
        output_features_dim,
        hidden_dims,
        num_blocks,
        num_heads,
        splits,
        num_hops,
        sparse,
        use_edges_features,
    ):
        """Initialize the lightning module"""
        super().__init__()

        self.model = Denoiser(
            grid_lon=grid_lon,
            grid_lat=grid_lat,
            input_features_dim=input_features_dim,
            output_features_dim=output_features_dim,
            hidden_dims=hidden_dims,
            num_blocks=num_blocks,
            num_heads=num_heads,
            splits=splits,
            num_hops=num_hops,
            device=self.device,
            sparse=sparse,
            use_edges_features=use_edges_features,
        )

        self.criterion = WeightedMSELoss(
            grid_lat=torch.tensor(grid_lat).to(self.device),
            pressure_levels=torch.tensor(pressure_levels).to(self.device),
            num_atmospheric_features=len(atmospheric_features),
            single_features_weights=torch.tensor([1.0, 0.1, 0.1, 0.1, 0.1]).to(self.device),
        )

        self.learning_rate = learning_rate
        self.cosine_t_max = cosine_t_max

    def forward(self, corrupted_targets, prev_inputs, noise_levels):
        """Forward pass of Gencast"""
        return self.model(corrupted_targets, prev_inputs, noise_levels)

    def training_step(self, batch):
        """Single training step"""
        corrupted_targets, prev_inputs, noise_levels, target_residuals = batch

        preds = self.model(
            corrupted_targets=corrupted_targets,
            prev_inputs=prev_inputs,
            noise_levels=noise_levels,
        )
        loss = self.criterion(preds, noise_levels, target_residuals)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        """Optimizer configuration"""
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cosine_t_max)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
                "interval": "step",  # step means "batch" here, default: epoch
                "frequency": 1,  # default
            },
        }

    def plot_sample(self, prev_inputs, target_residuals):
        """Plot predicted 2m_temperature and geopotential for given samples."""
        prev_inputs = prev_inputs[:1, :, :, :]
        target = target_residuals[:1, :, :, :]
        sampler = Sampler()
        preds = sampler.sample(self.model, prev_inputs)

        fig1, ax = plt.subplots(2)
        ax[0].imshow(preds[0, :, :, 78].T.cpu(), origin="lower", cmap="RdBu", vmin=-5, vmax=5)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("Diffusion sampling prediction")

        ax[1].imshow(target[0, :, :, 78].T.cpu(), origin="lower", cmap="RdBu", vmin=-5, vmax=5)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title("Ground truth")

        fig2, ax = plt.subplots(2)
        ax[0].imshow(preds[0, :, :, 12].T.cpu(), origin="lower", cmap="RdBu", vmin=-5, vmax=5)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title("Diffusion sampling prediction")

        ax[1].imshow(target[0, :, :, 12].T.cpu(), origin="lower", cmap="RdBu", vmin=-5, vmax=5)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title("Ground truth")
        return fig1, fig2


class SamplingCallback(Callback):  # noqa: D101
    def __init__(self, data):  # noqa: D107
        _, prev_inputs, _, target_residuals = data
        self.prev_inputs = torch.tensor(prev_inputs).unsqueeze(0)
        self.target_residuals = torch.tensor(target_residuals).unsqueeze(0)

    def on_train_epoch_start(self, trainer, pl_module):  # noqa: D102
        print("New epoch is starting")
        fig1, fig2 = pl_module.plot_sample(
            self.prev_inputs.to(pl_module.device), self.target_residuals.to(pl_module.device)
        )
        trainer.logger.log_image(
            key="samples", images=[fig1, fig2], caption=["2m_temperature", "geopotential"]
        )
        print("Uploaded samples")


if __name__ == "__main__":
    # define dataloader
    dataset = GenCastDataset(
        obs_path=OBS_PATH,
        atmospheric_features=atmospheric_features,
        single_features=single_features,
        static_features=static_features,
        max_year=2018,
        time_step=2,
    )

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        multiprocessing_context="forkserver",
    )

    # define/resume model

    num_steps = NUM_EPOCHS * len(dataloader) // NUM_DEVICES
    initial_lr = INITIAL_LR

    denoiser = LitModel.load_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        learning_rate=initial_lr,
        cosine_t_max=num_steps,
        pressure_levels=dataset.pressure_levels,
        grid_lon=dataset.grid_lon,
        grid_lat=dataset.grid_lat,
        input_features_dim=dataset.input_features_dim,
        output_features_dim=dataset.output_features_dim,
        **CFG,
    )

    # denoiser.model.load_state_dict(torch.load("model.pt", map_location = denoiser.device))
    # denoiser = torch.compile(denoiser)

    # define trainer
    wandb_logger = WandbLogger(project="gencast")
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=NUM_DEVICES,
        strategy="DDP",
        # precision=16,
        max_epochs=NUM_EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, SamplingCallback(data=dataset[0]), lr_monitor],
        log_every_n_steps=1,
    )

    # start training
    print("Starting training")
    trainer.fit(model=denoiser, train_dataloaders=dataloader)