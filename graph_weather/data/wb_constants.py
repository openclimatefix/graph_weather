from typing import Optional, List

import numpy as np
import xarray as xr


class WeatherBenchConstantFields:
    def __init__(
        self,
        const_fname: str,
        const_names: Optional[List[str]] = None,
        batch_chunk_size: int = 4,
    ) -> None:
        """
        Args:
            const_fname: file name where the constant fields are stored
                         transformations: z is normalized, ...
            const_names: constant field names
        """
        # constant fields
        self._Xc = xr.load_dataset(const_fname)

        if const_names is not None:
            # retain only the constant fields we want
            self._Xc = self._Xc[const_names]

        if "z" in const_names:
            # normalize orography field
            mu_z, sd_z = self._Xc["z"].mean(), self._Xc["z"].std()
            self._Xc["z"] = (self._Xc["z"] - mu_z) / sd_z

        self._constants = np.stack([self._Xc[var].values for var in const_names], axis=-1)

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

        self._constants = np.concatenate([self._constants, self.X_latlon], axis=-1)
        self._constants = np.stack([self._constants for _ in range(batch_chunk_size)], axis=0)  # batch axis

        # reshape the latlon info into what the GNN expects
        self._latlons = np.array([lats, lons]).T.reshape((-1, 2))

        # number of constant arrays
        self._nconst = self._constants.shape[-1]

    @property
    def constants(self):
        """
        Returns the constant data as a numpy array of shape (batch_chunk_size, lat, lon, nvar)
        """
        return self._constants

    @property
    def latlons(self):
        """
        Returns the lat-lon meshgrid data as a numpy array of shape (lat * lon, 2)
        """
        return self._latlons

    @property
    def nconst(self):
        return self._nconst
