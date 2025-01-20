"""
A custom PyTorch Dataset implementation for AMSU datasets.

This script defines a custom PyTorch Dataset (`AMSUDataset`) for working with AMSU datasets.
The dataset is loaded via the nnja library's `DataCatalog` and filtered for specific times and
variables. Each data point consists of a timestamp, latitude, longitude, and associated metadata.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from nnja import DataCatalog
except ImportError:
    print(
        "NNJA-AI library not installed. Please install with `pip install git+https://github.com/brightbandtech/nnja-ai.git`"
    )


class AMSUDataset(Dataset):
    """A custom PyTorch Dataset for handling AMSU data.

    This dataset retrieves observations and their metadata, filtered by the provided time and
    variable descriptors.
    """

    def __init__(self, dataset_name, time, primary_descriptors, additional_variables):
        """Initialize the AMSU dataset loader.

        Args:
            dataset_name: Name of the dataset to load.
            time: Specific timestamp to filter the data.
            primary_descriptors: List of primary descriptor variables to include (e.g.,
                               OBS_TIMESTAMP, LAT, LON).
            additional_variables: List of additional variables to include in metadata.
        """
        self.dataset_name = dataset_name
        self.time = time
        self.primary_descriptors = primary_descriptors
        self.additional_variables = additional_variables

        # Load data catalog and dataset
        self.catalog = DataCatalog(skip_manifest=True)
        self.dataset = self.catalog[self.dataset_name]
        self.dataset.load_manifest()

        self.dataset = self.dataset.sel(
            time=self.time, variables=self.primary_descriptors + self.additional_variables
        )
        self.dataframe = self.dataset.load_dataset(engine="pandas")

        for col in primary_descriptors:
            if col not in self.dataframe.columns:
                raise ValueError(f"The dataset must include a '{col}' column.")

        self.metadata_columns = [
            col for col in self.dataframe.columns if col not in self.primary_descriptors
        ]

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, index):
        """Return the observation and metadata for a given index.

        Args:
            index: Index of the observation to retrieve.

        Returns:
            A dictionary containing timestamp, latitude, longitude, and metadata.
        """
        row = self.dataframe.iloc[index]
        time = row["OBS_TIMESTAMP"].timestamp()
        latitude = row["LAT"]
        longitude = row["LON"]
        metadata = np.array([row[col] for col in self.metadata_columns], dtype=np.float32)

        return {
            "timestamp": torch.tensor(time, dtype=torch.float32),
            "latitude": torch.tensor(latitude, dtype=torch.float32),
            "longitude": torch.tensor(longitude, dtype=torch.float32),
            "metadata": torch.from_numpy(metadata),
        }


def collate_fn(batch):
    """Custom collate function to handle batching of dictionary data.

    Args:
        batch: List of dictionaries from __getitem__

    Returns:
        Single dictionary with batched tensors
    """
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0].keys()}


if __name__ == "__main__":
    # Configuration
    dataset_name = "amsua-1bamua-NC021023"
    time = "2021-01-01 00Z"
    primary_descriptors = ["OBS_TIMESTAMP", "LAT", "LON"]
    additional_variables = ["TMBR_00001"]

    # Initialize dataset
    amsu_dataset = AMSUDataset(dataset_name, time, primary_descriptors, additional_variables)

    batch_size = 4
    data_loader = DataLoader(
        amsu_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Example usage with batched data
    for batch in data_loader:
        print(f"Batch size: {batch['timestamp'].shape[0]}")
        print("Timestamps shape:", batch["timestamp"].shape)
        print("Latitudes shape:", batch["latitude"].shape)
        print("Longitudes shape:", batch["longitude"].shape)
        print("Metadata shape:", batch["metadata"].shape)

        for i in range(batch_size):
            print(f"\nItem {i}:")
            print("Time:", batch["timestamp"][i].item())
            print("Latitude:", batch["latitude"][i].item())
            print("Longitude:", batch["longitude"][i].item())
            print("Metadata:", batch["metadata"][i])

        break
