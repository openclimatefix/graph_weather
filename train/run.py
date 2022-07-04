"""Training script for training the weather forecasting model"""
import json

import datasets
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import xarray as xr
from datasets import Array2D, Array3D, Features, Sequence, Value
from pysolar.util import extraterrestrial_irrad
from torch.utils.data import DataLoader, Dataset, IterableDataset

from graph_weather import GraphWeatherForecaster
from graph_weather.data import const
from graph_weather.models.losses import NormalizedMSELoss

const.FORECAST_MEANS = {var: np.asarray(value) for var, value in const.FORECAST_MEANS.items()}
const.FORECAST_STD = {var: np.asarray(value) for var, value in const.FORECAST_STD.items()}


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_mean_stds():
    names = [
        "CLMR",
        "GRLE",
        "VVEL",
        "VGRD",
        "UGRD",
        "O3MR",
        "CAPE",
        "TMP",
        "PLPL",
        "DZDT",
        "CIN",
        "HGT",
        "RH",
        "ICMR",
        "SNMR",
        "SPFH",
        "RWMR",
        "TCDC",
        "ABSV",
    ]
    means = {}
    stds = {}
    # For pressure level values
    for n in names:
        if (
            len(
                sorted(
                    [
                        float(var.split(".", 1)[-1].split("_")[0])
                        for var in const.FORECAST_MEANS
                        if "mb" in var and n in var and "-" not in var
                    ]
                )
            )
            > 0
        ):
            means[n + "_mb"] = []
            stds[n + "_mb"] = []
            for value in sorted(
                [
                    float(var.split(".", 1)[-1].split("_")[0])
                    for var in const.FORECAST_MEANS
                    if "mb" in var and n in var and "-" not in var
                ]
            ):
                # Is floats now, but will be fixed
                if value >= 1:
                    value = int(value)
                var_name = f"{n}.{value}_mb"
                # print(var_name)

                means[n + "_mb"].append(const.FORECAST_MEANS[var_name])
                stds[n + "_mb"].append(const.FORECAST_STD[var_name])
            means[n + "_mb"] = np.mean(np.stack(means[n + "_mb"], axis=-1))
            stds[n + "_mb"] = np.mean(np.stack(stds[n + "_mb"], axis=-1))

    # For surface values
    for n in list(
        set(
            [
                var.split(".", 1)[0]
                for var in const.FORECAST_MEANS
                if "surface" in var
                and "level" not in var
                and "2e06" not in var
                and "below" not in var
                and "atmos" not in var
                and "tropo" not in var
                and "iso" not in var
                and "planetary_boundary_layer" not in var
            ]
        )
    ):
        means[n] = const.FORECAST_MEANS[n + ".surface"]
        stds[n] = const.FORECAST_STD[n + ".surface"]

    # For Cloud levels
    for n in list(
        set(
            [
                var.split(".", 1)[0]
                for var in const.FORECAST_MEANS
                if "sigma" not in var
                and "level" not in var
                and "2e06" not in var
                and "below" not in var
                and "atmos" not in var
                and "tropo" not in var
                and "iso" not in var
                and "planetary_boundary_layer" not in var
            ]
        )
    ):
        if "LCDC" in n:  # or "MCDC" in n or "HCDC" in n:
            means[n] = const.FORECAST_MEANS["LCDC.low_cloud_layer"]
            stds[n] = const.FORECAST_STD["LCDC.low_cloud_layer"]
        if "MCDC" in n:  # or "HCDC" in n:
            means[n] = const.FORECAST_MEANS["MCDC.middle_cloud_layer"]
            stds[n] = const.FORECAST_STD["MCDC.middle_cloud_layer"]
        if "HCDC" in n:
            means[n] = const.FORECAST_MEANS["HCDC.high_cloud_layer"]
            stds[n] = const.FORECAST_STD["HCDC.high_cloud_layer"]

    # Now for each of these
    means["max_wind"] = []
    stds["max_wind"] = []
    for n in sorted([var for var in const.FORECAST_MEANS if "max_wind" in var]):
        means["max_wind"].append(const.FORECAST_MEANS[n])
        stds["max_wind"].append(const.FORECAST_STD[n])
    means["max_wind"] = np.stack(means["max_wind"], axis=-1)
    stds["max_wind"] = np.stack(stds["max_wind"], axis=-1)

    for i in [2, 10, 20, 30, 40, 50, 80, 100]:
        means[f"{i}m_above_ground"] = []
        stds[f"{i}m_above_ground"] = []
        for n in sorted([var for var in const.FORECAST_MEANS if f"{i}_m_above_ground" in var]):
            means[f"{i}m_above_ground"].append(const.FORECAST_MEANS[n])
            stds[f"{i}m_above_ground"].append(const.FORECAST_STD[n])
        means[f"{i}m_above_ground"] = np.stack(means[f"{i}m_above_ground"], axis=-1)
        stds[f"{i}m_above_ground"] = np.stack(stds[f"{i}m_above_ground"], axis=-1)
    return means, stds


class XrDataset(IterableDataset):
    def __init__(self, resolution="2.0deg"):
        super().__init__()
        if "2deg" in resolution:
            LATITUDE = 91
            LONGITUDE = 180
        elif "1deg" in resolution:
            LATITUDE = 181
            LONGITUDE = 360
        elif "0.5deg" in resolution:
            LATITUDE = 361
            LONGITUDE = 720
        else:
            LATITUDE = 721
            LONGITUDE = 1440
        features = Features(
            {
                "next_CLMR_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "current_CLMR_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "CLMR_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_GRLE_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "current_GRLE_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "GRLE_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_VVEL_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "current_VVEL_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "VVEL_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_VGRD_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "current_VGRD_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "VGRD_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_UGRD_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "current_UGRD_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "UGRD_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_O3MR_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "current_O3MR_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "O3MR_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_TMP_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "current_TMP_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "TMP_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_DZDT_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "current_DZDT_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "DZDT_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_HGT_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "current_HGT_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "HGT_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_RH_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "current_RH_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "RH_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_ICMR_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "current_ICMR_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "ICMR_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_SNMR_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "current_SNMR_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "SNMR_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_SPFH_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "current_SPFH_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "SPFH_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_RWMR_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "current_RWMR_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "RWMR_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_TCDC_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "current_TCDC_mb": Array3D(shape=(LATITUDE, LONGITUDE, 22), dtype="float32"),
                "TCDC_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_ABSV_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "current_ABSV_mb": Array3D(shape=(LATITUDE, LONGITUDE, 41), dtype="float32"),
                "ABSV_mb_levels": Sequence(feature=Value(dtype="float32"), length=-1),
                "next_LFTX": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_LFTX": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_CRAIN": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_CRAIN": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_HGT": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_HGT": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_TMP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_TMP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_VIS": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_VIS": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_FRICV": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_FRICV": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_PRES": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_PRES": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_CAPE": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_CAPE": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_CFRZR": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_CFRZR": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_CNWAT": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_CNWAT": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_SNOD": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_SNOD": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_ICETK": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_ICETK": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_CIN": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_CIN": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_FLDCP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_FLDCP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_WEASD": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_WEASD": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_ICEC": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_ICEC": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_PRATE": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_PRATE": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_SUNSD": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_SUNSD": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_LAND": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_LAND": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_4LFTX": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_4LFTX": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_SFCR": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_SFCR": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_CSNOW": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_CSNOW": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_HPBL": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_HPBL": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_CICEP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_CICEP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_GUST": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_GUST": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_WILT": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_WILT": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_CPOFP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_CPOFP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_SOTYP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_SOTYP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_ICETMP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_ICETMP": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_VEG": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_VEG": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_HINDEX": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_HINDEX": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_MCDC": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_MCDC": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_LCDC": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_LCDC": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_HCDC": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "current_HCDC": Array2D(shape=(LATITUDE, LONGITUDE), dtype="float32"),
                "next_max_wind": Array3D(shape=(LATITUDE, LONGITUDE, 6), dtype="float32"),
                "current_max_wind": Array3D(shape=(LATITUDE, LONGITUDE, 6), dtype="float32"),
                "next_2m_above_ground": Array3D(shape=(LATITUDE, LONGITUDE, 5), dtype="float32"),
                "current_2m_above_ground": Array3D(shape=(LATITUDE, LONGITUDE, 5), dtype="float32"),
                "next_10m_above_ground": Array3D(shape=(LATITUDE, LONGITUDE, 2), dtype="float32"),
                "current_10m_above_ground": Array3D(
                    shape=(LATITUDE, LONGITUDE, 2), dtype="float32"
                ),
                "next_20m_above_ground": Array3D(shape=(LATITUDE, LONGITUDE, 2), dtype="float32"),
                "current_20m_above_ground": Array3D(
                    shape=(LATITUDE, LONGITUDE, 2), dtype="float32"
                ),
                "next_30m_above_ground": Array3D(shape=(LATITUDE, LONGITUDE, 2), dtype="float32"),
                "current_30m_above_ground": Array3D(
                    shape=(LATITUDE, LONGITUDE, 2), dtype="float32"
                ),
                "next_40m_above_ground": Array3D(shape=(LATITUDE, LONGITUDE, 2), dtype="float32"),
                "current_40m_above_ground": Array3D(
                    shape=(LATITUDE, LONGITUDE, 2), dtype="float32"
                ),
                "next_50m_above_ground": Array3D(shape=(LATITUDE, LONGITUDE, 2), dtype="float32"),
                "current_50m_above_ground": Array3D(
                    shape=(LATITUDE, LONGITUDE, 2), dtype="float32"
                ),
                "next_80m_above_ground": Array3D(shape=(LATITUDE, LONGITUDE, 5), dtype="float32"),
                "current_80m_above_ground": Array3D(
                    shape=(LATITUDE, LONGITUDE, 5), dtype="float32"
                ),
                "next_100m_above_ground": Array3D(shape=(LATITUDE, LONGITUDE, 3), dtype="float32"),
                "current_100m_above_ground": Array3D(
                    shape=(LATITUDE, LONGITUDE, 3), dtype="float32"
                ),
                "timestamps": Sequence(feature=Value(dtype="timestamp[ns]"), length=-1),
                "reftime": Value(dtype="timestamp[ns]"),
                "latitude": Sequence(feature=Value(dtype="float32"), length=-1),
                "longitude": Sequence(feature=Value(dtype="float32"), length=-1),
            }
        )
        self.dataset = datasets.load_dataset(
            "openclimatefix/gfs-surface-pressure-2.0deg",
            split="train",
            streaming=True,
            features=features,
        )
        self.means, self.stds = get_mean_stds()
        self.landsea = xr.open_zarr("/home/bieker/Downloads/landsea.zarr", consolidated=True).load()
        self.landsea_fixed = None

    def __iter__(self):
        self.dataset = self.dataset.shuffle(
            seed=np.random.randint(low=-1000, high=10000), buffer_size=4
        )
        for data in iter(self.dataset):
            # TODO Currently leaves out lat/lon/Sun irradience, and land/sea mask and topographic data
            data.update(
                {
                    key: np.expand_dims(np.asarray(value), axis=-1)
                    for key, value in data.items()
                    if key.replace("current_", "").replace("next_", "") in self.means.keys()
                    and np.asarray(value).ndim == 2
                }
            )  # Add third dimension for ones with 2
            input_data = {
                key.replace("current_", ""): torch.from_numpy(
                    (value - self.means[key.replace("current_", "")])
                    / self.stds[key.replace("current_", "")]
                )
                for key, value in data.items()
                if "current" in key and "time" not in key
            }
            output_data = {
                key.replace("next_", ""): torch.from_numpy(
                    (value - self.means[key.replace("next_", "")])
                    / self.stds[key.replace("next_", "")]
                )
                for key, value in data.items()
                if "next" in key and "time" not in key
            }
            # Stack them now
            # Add in the lat/lon coordinates
            # Add in the solar irradience
            if self.landsea_fixed is None:
                # Land-sea mask data, resampled to the same as the physical variables
                landsea = self.landsea.interp(
                    latitude=np.asarray(data["latitude"]).flatten()
                ).interp(longitude=np.asarray(data["longitude"]).flatten())
                # Calculate sin,cos, day of year, solar irradiance here before stacking
                landsea = np.stack(
                    [
                        (landsea[f"{var}"].values - const.LANDSEA_MEAN[var])
                        / const.LANDSEA_STD[var]
                        for var in landsea.data_vars
                        if not np.isnan(landsea[f"{var}"].values).any()
                    ],
                    axis=-1,
                )
                self.landsea_fixed = torch.from_numpy(landsea.T.reshape((-1, landsea.shape[-1])))

            lat_lons = np.array(
                np.meshgrid(
                    np.asarray(data["latitude"]).flatten(), np.asarray(data["longitude"]).flatten()
                )
            ).T.reshape((-1, 2))
            sin_lat_lons = np.sin(lat_lons)
            cos_lat_lons = np.cos(lat_lons)
            date = pd.to_datetime(data["timestamps"][0], utc=True)

            solar_times = [
                np.array(
                    [
                        extraterrestrial_irrad(
                            when=date.to_pydatetime(), latitude_deg=lat, longitude_deg=lon
                        )
                        for lat, lon in lat_lons
                    ]
                )
            ]
            for when in pd.date_range(
                date - pd.Timedelta("12 hours"), date + pd.Timedelta("12 hours"), freq=f"1H"
            ):
                solar_times.append(
                    np.array(
                        [
                            extraterrestrial_irrad(
                                when=when.to_pydatetime(), latitude_deg=lat, longitude_deg=lon
                            )
                            for lat, lon in lat_lons
                        ]
                    )
                )
            solar_times = np.array(solar_times)

            # Normalize to between -1 and 1
            solar_times -= const.SOLAR_MEAN
            solar_times /= const.SOLAR_STD

            input_data = torch.concat([value for _, value in input_data.items()], dim=-1)
            output_data = torch.concat([value for _, value in output_data.items()], dim=-1)
            input_data = input_data.transpose(0, 1).reshape(-1, input_data.shape[-1])
            output_data = output_data.transpose(0, 1).reshape(-1, input_data.shape[-1])
            day_of_year = data["timestamps"][0].dayofyear / 366.0
            sin_of_year = np.ones_like(lat_lons)[:, 0] * np.sin(day_of_year)
            cos_of_year = np.ones_like(lat_lons)[:, 0] * np.cos(day_of_year)
            to_concat = [
                input_data,
                torch.permute(torch.from_numpy(solar_times), (1, 0)),
                torch.from_numpy(sin_lat_lons),
                torch.from_numpy(cos_lat_lons),
                torch.from_numpy(np.expand_dims(sin_of_year, axis=-1)),
                torch.from_numpy(np.expand_dims(cos_of_year, axis=-1)),
                self.landsea_fixed,
            ]
            input_data = torch.concat(to_concat, dim=-1)
            yield input_data, output_data


# print("Done coarsening")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Get the variance of the variables
hf_ds = datasets.load_dataset(
    "openclimatefix/gfs-surface-pressure-2.0deg", split="train", streaming=True
)
data = next(iter(hf_ds))
lat_lons = np.array(
    np.meshgrid(np.asarray(data["latitude"]).flatten(), np.asarray(data["longitude"]).flatten())
).T.reshape(-1, 2)
dset = XrDataset()
dataset = DataLoader(dset, batch_size=1, num_workers=8, worker_init_fn=worker_init_fn)
feature_variances = []
for var in range(605):
    feature_variances.append(0.0)
criterion = NormalizedMSELoss(
    lat_lons=lat_lons, feature_variance=feature_variances, device=device
).to(device)
means = []
model = GraphWeatherForecaster(
    lat_lons,
    edge_dim=1024,
    hidden_dim_processor_edge=1024,
    node_dim=1024,
    hidden_dim_processor_node=1024,
    hidden_dim_decoder=1024,
    feature_dim=605,
    aux_dim=40,
    num_blocks=6,
).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
print("Done Setup")
import time

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    print(f"Start Epoch: {epoch}")
    for i, data in enumerate(dataset):
        start = time.time()
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
        if i % 10 == 0:
            assert not np.isnan(running_loss)
            model.push_to_hub(
                "graph-weather-forecaster-2.0deg",
                organization="openclimatefix",
                commit_message=f"Add model Epoch={epoch}, i={i}",
            )
    if epoch % 5 == 0:
        assert not np.isnan(running_loss)
        model.push_to_hub(
            "graph-weather-forecaster-2.0deg",
            organization="openclimatefix",
            commit_message=f"Add model Epoch={epoch}",
        )

print("Finished Training")
