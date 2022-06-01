"""Training script for training the weather forecasting model"""
from graph_weather import GraphWeatherForecaster
import xarray as xr
from torch.utils.data import DataLoader, Dataset
import numpy as np
from graph_weather.models.losses import NormalizedMSELoss
import torchvision.transforms as transforms
import torch.optim as optim
import torch


class XrDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.filepaths = [
            "/run/media/bieker/LargeSSD/gdas1.fnl0p25.2016010100.f00.zarr",
            "/run/media/bieker/LargeSSD/gdas1.fnl0p25.2016010106.f00.zarr",
            "/run/media/bieker/LargeSSD/gdas1.fnl0p25.2016010112.f00.zarr",
            "/run/media/bieker/LargeSSD/gdas1.fnl0p25.2016010118.f00.zarr",
        ]

    def __len__(self):
        return len(self.filepaths) - 1

    def __getitem__(self, item):
        start = (
            xr.open_zarr(self.filepaths[item], consolidated=True)
            .coarsen(latitude=8, boundary="pad")
            .mean()
            .coarsen(longitude=8)
            .mean()
        )
        end = (
            xr.open_zarr(self.filepaths[item + 1], consolidated=True)
            .coarsen(latitude=8, boundary="pad")
            .mean()
            .coarsen(longitude=8)
            .mean()
        )

        # Stack the data into a large data cube
        input_data = np.stack(
            [
                start[f"{var}"].values
                for var in start.data_vars
                if not np.isnan(start[f"{var}"].values).any()
            ],
            axis=-1,
        )
        assert not np.isnan(input_data).any()
        mean = np.mean(input_data, axis=(0, 1))
        std = np.std(input_data, axis=(0, 1))
        output_data = np.stack(
            [
                end[f"{var}"].values
                for var in end.data_vars
                if not np.isnan(end[f"{var}"].values).any()
            ],
            axis=-1,
        )
        assert not np.isnan(output_data).any()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        # Normalize now
        return (
            transform(input_data).transpose(0, 1).reshape(-1, input_data.shape[-1]),
            transform(output_data).transpose(0, 1).reshape(-1, input_data.shape[-1]),
        )


dataset = DataLoader(XrDataset(), batch_size=1, num_workers=3)
data = (
    xr.open_zarr("/run/media/bieker/LargeSSD/gdas1.fnl0p25.2016010100.f00.zarr", consolidated=True)
    .coarsen(latitude=8, boundary="pad")
    .mean()
    .coarsen(longitude=8)
    .mean()
)
lat_lons = np.array(np.meshgrid(data.latitude.values, data.longitude.values)).T.reshape(-1, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = NormalizedMSELoss(
    lat_lons=lat_lons, feature_variance=np.ones((289,)), device=device
).to(device)
model = GraphWeatherForecaster(lat_lons, feature_dim=289, device=device, num_blocks=6).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(dataset, 0):
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
        print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.3f}")

print("Finished Training")
