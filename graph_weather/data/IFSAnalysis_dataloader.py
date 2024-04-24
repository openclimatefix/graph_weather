import numpy as np
import torchvision.transforms as transforms
import xarray as xr
from torch.utils.data import Dataset

IFS_MEAN = {
    "geopotential": 78054.78,
    "specific_humidity": 0.0018220816,
    "temperature": 243.41727,
    "u_component_of_wind": 7.3073797,
    "v_component_of_wind": 0.032221083,
    "vertical_velocity": 0.0058287205,
}

IFS_STD = {
    "geopotential": 59538.875,
    "specific_humidity": 0.0035489395,
    "temperature": 29.211119,
    "u_component_of_wind": 13.777036,
    "v_component_of_wind": 8.867598,
    "vertical_velocity": 0.08577341,
}


class IFSAnalisysDataset(Dataset):
    def __init__(self, filepath: str, features: list, start_year: int = 2016, end_year: int = 2022):
        super().__init__()
        assert (
            start_year <= end_year
        ), f"start_year ({start_year}) cannot be greater than end_year ({end_year})."
        assert start_year >= 2016 and start_year <= 2022, "Time data range from 2016 to 2022"
        assert end_year >= 2016 and end_year <= 2022, "Time data range from 2016 to 2022"
        self.data = xr.open_zarr(filepath)
        self.data = self.data.sel(
            time=slice(str(start_year), str(end_year))
        )  # Filter data by start and end years

        self.NWP_features = features

    def __len__(self):
        return len(self.data["time"])

    def __getitem__(self, idx):

        start = self.data.isel(time=idx)
        end = self.data.isel(time=idx + 1)

        # Extract NWP features
        input_data = self.nwp_features_extraction(start)
        output_data = self.nwp_features_extraction(end)

        return (
            (transforms).ToTensor()(input_data).view(-1, input_data.shape[-1]),
            (transforms).ToTensor()(output_data).view(-1, output_data.shape[-1]),
        )

    def nwp_features_extraction(self, data):
        data_cube = np.stack(
            [
                (data[f"{var}"].values - IFS_MEAN[f"{var}"]) / (IFS_STD[f"{var}"] + 1e-6)
                for var in self.NWP_features
            ],
            axis=-1,
        ).astype(np.float32)

        num_layers, num_lat, num_lon, num_vars = data_cube.shape
        data_cube = data_cube.reshape(num_lat, num_lon, num_vars * num_layers)

        assert not np.isnan(data_cube).any()
        return data_cube
