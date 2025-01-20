import numpy as np
import pandas as pd
from nnja.io import _check_authentication
from torch.utils.data import Dataset, DataLoader

if _check_authentication():
    from nnja import DataCatalog

    class AMSUDataset(Dataset):
        def __init__(self, dataset_name, time, primary_descriptors, additional_variables):
            """
            Initialize the AMSU dataset loader.
            :param dataset_name: Name of the dataset to load.
            :param time: Specific timestamp to filter the data.
            :param primary_descriptors: List of primary descriptor variables to include (e.g., OBS_TIMESTAMP, LAT, LON).
            :param additional_variables: List of additional variables to include in metadata.
            """
            self.dataset_name = dataset_name
            self.time = time
            self.primary_descriptors = primary_descriptors
            self.additional_variables = additional_variables

            # Load data catalog and dataset
            self.catalog = DataCatalog(skip_manifest=True)
            self.dataset = self.catalog[self.dataset_name]
            self.dataset.load_manifest()

            self.dataset = self.dataset.sel(time=self.time, variables=self.primary_descriptors + self.additional_variables)
            self.dataframe = self.dataset.load_dataset(engine='pandas')

            for col in primary_descriptors:
                if col not in self.dataframe.columns:
                    raise ValueError(f"The dataset must include a '{col}' column.")

            self.metadata_columns = [
                col for col in self.dataframe.columns if col not in self.primary_descriptors
            ]

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            """
            Returns the observation and metadata for a given index.
            :param index: Index of the observation to retrieve.
            :return: A tuple (time, latitude, longitude, metadata).
            """
            row = self.dataframe.iloc[index]
            time = row["OBS_TIMESTAMP"].timestamp()  
            latitude = row["LAT"]
            longitude = row["LON"]
            metadata = np.array([row[col] for col in self.metadata_columns], dtype=np.float32)
            return time, latitude, longitude, metadata

    # Configuration
    dataset_name = "amsua-1bamua-NC021023"
    time = "2021-01-01 00Z"
    primary_descriptors = ["OBS_TIMESTAMP", "LAT", "LON"]
    additional_variables = ["TMBR_00001"]

    # Initialize dataset
    amsu_dataset = AMSUDataset(dataset_name, time, primary_descriptors, additional_variables)

    # Use DataLoader without batching
    data_loader = DataLoader(amsu_dataset, shuffle=True)

    # Example usage
    for time, latitude, longitude, metadata in data_loader:
        print("Time:", time)
        print("Latitude:", latitude)
        print("Longitude:", longitude)
        print("Metadata:", metadata)
else:
    print("Install nnjai lib. pip install git+https://github.com/brightbandtech/nnja-ai.git")
