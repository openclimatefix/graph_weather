"""Lightning-based training script for any WeatherModel on Icechunk repositories.

Trains a WeatherModel (with any encoder/processor/decoder combo) on data
loaded from Icechunk repositories using PyTorch Lightning.
"""

import argparse
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from zarr import open_group

from graph_weather.models import StageConfig, WeatherModel, build_graphcast_stage


class IcechunkDataset(Dataset):
    """Dataset that loads data from an Icechunk (Zarr) repository.

    Expects the Zarr store to contain 'input' and 'target' arrays of shape
    [num_samples, num_points, num_features].
    """

    def __init__(self, repo_path: str, split: str = "train", max_samples: Optional[int] = None):
        """
        Initialize the dataset.

        Args:
            repo_path: Path to the Icechunk/Zarr repository.
            split: Data split to use ('train', 'val', 'test').
            max_samples: Maximum number of samples to load (for debugging).
        """
        self.store = open_group(repo_path, mode="r")
        self.split = split

        # Find the correct group for this split
        if split in self.store:
            self.group = self.store[split]
        else:
            self.group = self.store

        # Load input and target data
        self.inputs = self.group["input"][:]
        self.targets = self.group["target"][:]

        if max_samples is not None:
            self.inputs = self.inputs[:max_samples]
            self.targets = self.targets[:max_samples]

        self.num_points = self.inputs.shape[1]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        """Return a PyG Data object for a single sample."""
        x = torch.from_numpy(self.inputs[idx]).float()  # [N, F]
        y = torch.from_numpy(self.targets[idx]).float()  # [N, F]
        data = Data(x=x, y=y, batch_size=1)
        return data


class WeatherModelLightning(pl.LightningModule):
    """Lightning module wrapping any WeatherModel instance."""

    def __init__(
        self,
        model: WeatherModel,
        learning_rate: float = 1e-3,
        loss_fn: Optional[torch.nn.Module] = None,
    ):
        """
        Initialize the Lightning module.

        Args:
            model: A WeatherModel instance.
            learning_rate: Learning rate for the optimizer.
            loss_fn: Loss function. Defaults to MSELoss.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()

    def forward(self, data: Data) -> Data:
        return self.model(data)

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Single training step."""
        # Combine batch samples into a single Data object
        x = batch.x  # [B*N, F]
        y = batch.y  # [B*N, F]
        batch_size = batch.batch_size if hasattr(batch, "batch_size") else 1

        # Create input data with batch_size
        input_data = Data(x=x, batch_size=batch_size)
        output_data = self.model(input_data)

        loss = self.loss_fn(output_data.x, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        """Single validation step."""
        x = batch.x
        y = batch.y
        batch_size = batch.batch_size if hasattr(batch, "batch_size") else 1

        input_data = Data(x=x, batch_size=batch_size)
        output_data = self.model(input_data)

        loss = self.loss_fn(output_data.x, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def collate_fn(batch: list) -> Data:
    """Collate a batch of PyG Data objects into a single batched Data object."""
    x_list = [item.x for item in batch]
    y_list = [item.y for item in batch]
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    batch_size = len(batch)
    return Data(x=x, y=y, batch_size=batch_size)


def build_model(args: argparse.Namespace) -> WeatherModel:
    """Build a WeatherModel from CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A WeatherModel instance.
    """
    # Build lat/lon grid
    lat_lons = []
    for lat in range(-90, 90, args.grid_resolution):
        for lon in range(0, 360, args.grid_resolution):
            lat_lons.append((lat, lon))

    if args.model_type == "graphcast":
        encoder, processor, decoder = build_graphcast_stage(
            lat_lons=lat_lons,
            resolution=args.h3_resolution,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            num_processor_blocks=args.processor_blocks,
            hidden_layers=args.hidden_layers,
            use_checkpointing=args.use_checkpointing,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    stage = StageConfig(
        encoder,
        processor,
        decoder,
        checkpoint_encoder=args.checkpoint_encoder,
        checkpoint_processor=args.checkpoint_processor,
        checkpoint_decoder=args.checkpoint_decoder,
    )

    return WeatherModel(stage, checkpoint_model=args.checkpoint_model)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a WeatherModel on Icechunk data")

    # Data arguments
    parser.add_argument(
        "--repo-path", type=str, required=True, help="Path to the Icechunk/Zarr repository"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples per split (for debugging)"
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="graphcast",
        choices=["graphcast"],
        help="Model architecture type",
    )
    parser.add_argument("--input-dim", type=int, default=78, help="Input feature dimension")
    parser.add_argument("--output-dim", type=int, default=78, help="Output feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument(
        "--processor-blocks", type=int, default=9, help="Number of processor blocks"
    )
    parser.add_argument("--hidden-layers", type=int, default=2, help="MLP hidden layers")
    parser.add_argument(
        "--grid-resolution", type=int, default=5, help="Grid resolution in degrees"
    )
    parser.add_argument("--h3-resolution", type=int, default=2, help="H3 resolution level")

    # Checkpointing arguments
    parser.add_argument(
        "--use-checkpointing", action="store_true", help="Enable fine-grained checkpointing"
    )
    parser.add_argument(
        "--checkpoint-model", action="store_true", help="Checkpoint entire model"
    )
    parser.add_argument(
        "--checkpoint-encoder", action="store_true", help="Checkpoint encoder"
    )
    parser.add_argument(
        "--checkpoint-processor", action="store_true", help="Checkpoint processor"
    )
    parser.add_argument(
        "--checkpoint-decoder", action="store_true", help="Checkpoint decoder"
    )

    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use")
    parser.add_argument(
        "--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu"],
        help="Accelerator type",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory"
    )

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Build model
    model = build_model(args)
    lightning_module = WeatherModelLightning(model, learning_rate=args.lr)

    # Build datasets
    train_dataset = IcechunkDataset(
        args.repo_path, split="train", max_samples=args.max_samples
    )
    val_dataset = IcechunkDataset(
        args.repo_path, split="val", max_samples=args.max_samples
    )

    # Build data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Checkpointing callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="weather-model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(lightning_module, train_loader, val_loader)
    print(f"Training complete. Best checkpoints saved to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
