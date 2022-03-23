"""

The dataloader has to do a few things for the model to work correctly

1. Load the land-0sea mask, orography dataset, regridded from 0.1 to the correct resolution
2. Calculate the top-of-atmosphere solar radiation for each location at fcurrent time and 10 other
 times +- 12 hours
3. Add day-of-year, sin(lat), cos(lat), sin(lon), cos(lon) as well
3. Batch data as either in geometric batches, or more normally
4. Rescale between 0 and 1, but don't normalize

"""

from pysolar.util import extraterrestrial_irrad
import xarray as xr
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import const


class AnalysisDataset(Dataset):
    def __init__(self, filepaths, invariant_path, mean, std, coarsen: int = 8):
        super().__init__()
        self.filepaths = sorted(filepaths)
        self.invariant_path = invariant_path
        self.coarsen = coarsen
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.filepaths) - 1

    def __getitem__(self, item):
        if self.coarsen <= 1:  # Don't coarsen, so don't even call it
            start = xr.open_zarr(self.filepaths[item], consolidated=True)
            end = xr.open_zarr(self.filepaths[item + 1], consolidated=True)
        else:
            start = (
                xr.open_zarr(self.filepaths[item], consolidated=True)
                .coarsen(latitude=self.coarsen, boundary="pad")
                .mean()
                .coarsen(longitude=self.coarsen)
                .mean()
            )
            end = (
                xr.open_zarr(self.filepaths[item + 1], consolidated=True)
                .coarsen(latitude=self.coarsen, boundary="pad")
                .mean()
                .coarsen(longitude=self.coarsen)
                .mean()
            )

        # Land-sea mask data, resampled to the same as the physical variables
        landsea = (
            xr.open_zarr(self.invariant_path, consolidated=True)
            .interp(latitude=start.latitude.values)
            .interp(longitude=start.longitude.values)
        )
        # Calculate sin,cos, day of year, solar irradiance here before stacking
        landsea = np.stack(
            [
                (landsea[f"{var}"].values - const.LANDSEA_MEAN[var]) / const.LANDSEA_STD[var]
                for var in landsea.data_vars
                if not np.isnan(landsea[f"{var}"].values).any()
            ],
            axis=-1,
        )
        landsea = landsea.T.reshape((-1, landsea.shape[-1]))
        lat_lons = np.array(np.meshgrid(start.latitude.values, start.longitude.values)).T.reshape(
            (-1, 2)
        )
        sin_lat_lons = np.sin(lat_lons)
        cos_lat_lons = np.cos(lat_lons)
        date = start.time.dt.values
        day_of_year = start.time.dayofyear.values / 365.0
        sin_of_year = np.sin(day_of_year)
        cos_of_year = np.cos(day_of_year)
        solar_times = [np.array([extraterrestrial_irrad(date, lat, lon) for lat, lon in lat_lons])]
        for when in pd.date_range(
            date - pd.Timedelta("12 hours"), date + pd.Timedelta("12 hours"), freq=f"1H"
        ):
            solar_times.append(
                np.array([extraterrestrial_irrad(when, lat, lon) for lat, lon in lat_lons])
            )
        solar_times = np.array(solar_times)

        # End time solar radiation too
        end_date = end.time.dt.values
        end_solar_times = [
            np.array([extraterrestrial_irrad(end_date, lat, lon) for lat, lon in lat_lons])
        ]
        for when in pd.date_range(
            end_date - pd.Timedelta("12 hours"), end_date + pd.Timedelta("12 hours"), freq=f"1H"
        ):
            end_solar_times.append(
                np.array([extraterrestrial_irrad(when, lat, lon) for lat, lon in lat_lons])
            )
        end_solar_times = np.array(solar_times)

        # Normalize to between -1 and 1
        solar_times -= const.SOLAR_MEAN
        solar_times /= const.SOLAR_STD
        end_solar_times -= const.SOLAR_MEAN
        end_solar_times /= const.SOLAR_STD

        # Stack the data into a large data cube
        input_data = np.stack(
            [
                start[f"{var}"].values
                for var in start.data_vars
                if not np.isnan(start[f"{var}"].values).any()
            ],
            axis=-1,
        )
        # TODO Combine with above? And include sin/cos of day of year
        input_data = np.concatenate(
            [
                input_data.T.reshape((-1, input_data.shape[-1])),
                sin_lat_lons,
                cos_lat_lons,
                solar_times,
                landsea,
            ],
            axis=-1,
        )
        # Not want to predict non-physics variables -> Output only the data variables? Would be simpler, and just add in the new ones each time

        output_data = np.stack(
            [
                end[f"{var}"].values
                for var in end.data_vars
                if not np.isnan(end[f"{var}"].values).any()
            ],
            axis=-1,
        )

        output_data = np.concatenate(
            [
                output_data.T.reshape((-1, output_data.shape[-1])),
                sin_lat_lons,
                cos_lat_lons,
                end_solar_times,
                landsea,
            ],
            axis=-1,
        )
        # Stick with Numpy, don't tensor it, as just going from 0 to 1

        # Normalize now
        return input_data, output_data


obs_data = xr.open_zarr(
    "/home/jacob/Development/prepbufr.gdas.20160101.t00z.nr.48h.raw.zarr", consolidated=True
)
# TODO Embedding? These should stay consistent across all of the inputs, so can just load the values, not the strings?
# Should only take in the quality markers, observations, reported observation time relative to start point
# Observation errors, and background values, lat/lon/height/speed of observing thing
print(obs_data)
print(obs_data.hdr_inst_typ.values)
print(obs_data.hdr_irpt_typ.values)
print(obs_data.obs_qty_table.values)
print(obs_data.hdr_prpt_typ.values)
print(obs_data.hdr_sid_table.values)
print(obs_data.hdr_typ_table.values)
print(obs_data.obs_desc.values)
print(obs_data.data_vars.keys())
exit()
analysis_data = xr.open_zarr(
    "/home/jacob/Development/gdas1.fnl0p25.2016010100.f00.zarr", consolidated=True
)
print(analysis_data)
