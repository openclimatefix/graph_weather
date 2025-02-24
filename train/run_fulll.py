"""Training script for training the weather forecasting model"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from graph_weather import GraphWeatherForecaster
from graph_weather.data import const
from graph_weather.models.losses import NormalizedMSELoss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


class XrDataset(Dataset):
    """
    Dataset class for loading data from Hugging Face datasets.

    Attributes:
        filepaths : List of file paths to the data.
        data : Dataset containing the loaded data.

    Methods:
        __init__: Initialize the XrDataset object by loading data from Hugging Face datasets.
        __len__: Get the length of the dataset.
        __getitem__: Get an item from the dataset by index.
    """

    def __init__(self):
        """
        Initialize the XrDataset object by loading data from Hugging Face datasets.
        """
        super().__init__()
        with open("hf_forecasts.json", "r") as f:
            files = json.load(f)
        self.filepaths = [
            "zip:///::https://huggingface.co/datasets/openclimatefix/gfs-reforecast/resolve/main/"
            + f
            for f in files
        ]
        self.data = xr.open_mfdataset(
            self.filepaths, engine="zarr", concat_dim="reftime", combine="nested"
        ).sortby("reftime")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, item):
        start_idx = np.random.randint(0, 14)
        data = self.data.isel(reftime=item, time=slice(start_idx, start_idx + 2))

        start = data.isel(time=0)
        end = data.isel(time=1)
        # Stack the data into a large data cube
        input_data = np.stack(
            [
                (start[f"{var}"].values - const.FORECAST_MEANS[f"{var}"])
                / (const.FORECAST_STD[f"{var}"] + 0.0001)
                for var in start.data_vars
                if "mb" in var or "surface" in var
            ],
            axis=-1,
        )
        input_data = np.nan_to_num(input_data)
        assert not np.isnan(input_data).any()
        output_data = np.stack(
            [
                (end[f"{var}"].values - const.FORECAST_MEANS[f"{var}"])
                / (const.FORECAST_STD[f"{var}"] + 0.0001)
                for var in end.data_vars
                if "mb" in var or "surface" in var
            ],
            axis=-1,
        )
        output_data = np.nan_to_num(output_data)
        assert not np.isnan(output_data).any()
        transform = transforms.Compose([transforms.ToTensor()])
        # Normalize now
        return (
            transform(input_data).transpose(0, 1).reshape(-1, input_data.shape[-1]),
            transform(output_data).transpose(0, 1).reshape(-1, input_data.shape[-1]),
        )


with open("hf_forecasts.json", "r") as f:
    files = json.load(f)
files = [
    "zip:///::https://huggingface.co/datasets/openclimatefix/gfs-reforecast/resolve/main/" + f
    for f in files
]
data = (
    xr.open_zarr(files[0], consolidated=True).isel(time=0)
    # .coarsen(latitude=8, boundary="pad")
    # .mean()
    # .coarsen(longitude=8)
    # .mean()
)
print(data)
# print("Done coarsening")
lat_lons = np.array(np.meshgrid(data.latitude.values, data.longitude.values)).T.reshape(-1, 2)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Get the variance of the variables
feature_variances = []
for var in data.data_vars:
    if "mb" in var or "surface" in var:
        feature_variances.append(const.FORECAST_DIFF_STD[var] ** 2)
criterion = NormalizedMSELoss(
    lat_lons=lat_lons, feature_variance=feature_variances, device=device
).to(device)
means = []
dataset = DataLoader(XrDataset(), batch_size=1)
model = GraphWeatherForecaster(lat_lons, feature_dim=597, num_blocks=6).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
print("Done Setup")


for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    start = time.time()
    print(f"Start Epoch: {epoch}")
    for i, data in enumerate(dataset):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        end = time.time()
        print(
            f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i + 1):.3f} Time: {end - start} sec"
        )
    if epoch % 5 == 0:
        assert not np.isnan(running_loss)
        model.push_to_hub(
            "graph-weather-forecaster-2.0deg",
            organization="openclimatefix",
            commit_message=f"Add model Epoch={epoch}",
        )

print("Finished Training")
