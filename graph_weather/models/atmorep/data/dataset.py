import os

from torch.utils.data import Dataset

from graph_weather.models.atmorep.config import AtmoRepConfig


class ERA5Dataset(Dataset):
    def __init__(
        self,
        config: AtmoRepConfig,
        data_dir: str,
        fields=None,
        years=None,
        months=None,
        transform=None,
    ):
        """
        ERA5Dataset for handling atmospheric data.

        Args:
            config (AtmoRepConfig): Configuration object.
            data_dir (str): Directory containing the data and index file.
            fields (list, optional): List of fields to load. Defaults to config.input_fields.
            years (list, optional): List of years to consider. Defaults to range(1979, 2022).
            months (list, optional): List of months to consider. Defaults to 1-12.
            transform (callable, optional): Optional transform to apply on a sample.
        """
        self.config = config
        self.data_dir = data_dir
        self.fields = fields if fields is not None else config.input_fields
        self.transform = transform

        # Expect a data index file in the data directory
        index_file = os.path.join(data_dir, "data_index.txt")
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                # Read each line (file name) into the index
                self.data_index = f.read().splitlines()
        else:
            raise Exception("Data index file not found in data_dir.")

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        """
        Retrieve a sample by using the file name from the data index and loading the data.
        """
        file_name = self.data_index[idx]
        file_path = os.path.join(self.data_dir, file_name)
        data = self.load_file(file_path)

        if self.transform:
            data = self.transform(data)

        return data

    def load_file(self, file_path):
        """
        Dummy implementation of load_file.
        In a production system, this would load data from netCDF or another data format.
        For testing purposes, this function is patched to return dummy data.
        """
        return {}
