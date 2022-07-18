from typing import Optional, List

import numpy as np
import xarray as xr


class ConstantData:
    def __init__(
        self,
        const_fname: str,
        const_names: Optional[List[str]] = None,
        batch_chunk_size: int = 8,
    ) -> None:
        """
        Args:
            const_fname: file name where the constant fields are stored
                         transformations: z is normalized, ...
            const_names: constant field names
        """
        # constant fields
        self._Xc = xr.load_dataset(const_fname).isel(time=0).drop("time")

        if const_names is not None:
            # retain only the constant fields we want
            self._Xc = self._Xc[const_names]

        if "z" in const_names:
            # normalize orography field
            mu_z, sd_z = self._Xc["z"].mean(), self._Xc["z"].std()
            self._Xc["z"] = (self._Xc["z"] - mu_z) / sd_z

        self._constants = np.stack([self._Xc[var].values for var in const_names], axis=0)

        lats, lons = np.meshgrid(self._Xc.latitude.values, self._Xc.longitude.values)

        # sine / cosine of latitude
        self.X_latlon = np.stack(
            [
                np.sin(lats.T * np.pi / 180.0),
                np.cos(lats.T * np.pi / 180.0),
                np.sin(lons.T * np.pi / 180.0),
                np.cos(lons.T * np.pi / 180.0),
            ],
            axis=-1,  # stack along new axis
        )

        self._constants = np.concatenate([self._constants, self.X_latlon], axis=0)

        self._constants = np.stack([self._constants for _ in range(batch_chunk_size)], axis=0)  # batch axis

    def get_constants(self):
        return self._constants
