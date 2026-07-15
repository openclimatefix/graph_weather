"""Dataset for the stretched-grid forecaster with global and regional observations.

Samples one IFS grid point per coarse H3 cell (global coverage) plus a dense
regional crop inside a movable bounding box.  Returns ``(features, lat_lons,
target, bbox)`` shaped for ``StretchedForecaster.forward``.

``_sample_box`` and ``_extract_points`` mirror the same methods in
``RegionalDataset`` but differ in return shape (``_sample_box`` also returns the
bbox) and scope (``_extract_points`` drops the coarsen path).  Kept separate to
avoid modifying the merged ``regional_dataset`` module.
"""

import h3
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from graph_weather.data.regional_dataset import (
    CORE_SURFACE,
    CORE_SURFACE_MEAN,
    CORE_SURFACE_STD,
    DEFAULT_STORE,
    open_ifs_store,
)
from graph_weather.models.layers.stretched_mesh import build_variable_resolution_mesh


class StretchedDataset(Dataset):
    """Global + regional IFS samples for a variable-resolution mesh.

    Args:
        dataset: An open ``xarray.Dataset`` to sample from.  When ``None``, the
            store at ``store_url`` is opened.
        store_url: Source Cooperative ``<bucket>/<prefix>`` of the IFS store.
        variables: Surface variable names to stack into the feature dimension.
        coarse_res: H3 resolution for the global coarse mesh.
        fine_res: H3 resolution for the refined region.
        extent_deg: Side length of the square bounding box, in degrees.
        max_regional_points: Cap on observation points inside the region.
        seed: Base seed; sample ``idx`` uses ``seed + idx`` so boxes move.
        mean: Per-variable means for standardisation.
        std: Per-variable standard deviations for standardisation.
    """

    def __init__(
        self,
        dataset: xr.Dataset = None,
        store_url: str = DEFAULT_STORE,
        variables: list = None,
        coarse_res: int = 2,
        fine_res: int = 3,
        extent_deg: float = 20.0,
        max_regional_points: int = 2000,
        seed: int = 0,
        mean: dict = None,
        std: dict = None,
    ):
        """Open the source grid and precompute per-cell nearest grid indices."""
        super().__init__()
        self.data = dataset if dataset is not None else open_ifs_store(store_url)
        self.variables = variables if variables is not None else CORE_SURFACE
        self.coarse_res = coarse_res
        self.fine_res = fine_res
        self.extent_deg = extent_deg
        self.max_regional_points = max_regional_points
        self.seed = seed
        self.mean = mean if mean is not None else CORE_SURFACE_MEAN
        self.std = std if std is not None else CORE_SURFACE_STD
        self.lat = self.data["latitude"].values
        self.lon = self.data["longitude"].values

        # Precompute: for each coarse cell, the nearest IFS grid index.
        self._coarse_cells = sorted(h3.uncompact_cells(h3.get_res0_cells(), coarse_res))
        self._cell_grid_idx = {}
        for cell in self._coarse_cells:
            clat, clon = h3.cell_to_latlng(cell)
            lat_i = int(np.argmin(np.abs(self.lat - clat)))
            lon_i = int(np.argmin(np.abs(self.lon - clon)))
            self._cell_grid_idx[cell] = (lat_i, lon_i)

    def __len__(self) -> int:
        """Number of t -> t+1 sample pairs."""
        return int(self.data.sizes["time"]) - 1

    def _sample_box(self, rng):
        """Pick a movable bbox centre and return point grid indices and coords."""
        half = self.extent_deg / 2.0
        lat_c = rng.uniform(self.lat.min() + half, self.lat.max() - half)
        lon_c = rng.uniform(self.lon.min() + half, self.lon.max() - half)

        lat_idx = np.flatnonzero(np.abs(self.lat - lat_c) <= half)
        lon_idx = np.flatnonzero(np.abs(self.lon - lon_c) <= half)

        glat, glon = np.meshgrid(self.lat[lat_idx], self.lon[lon_idx], indexing="ij")
        giy, gix = np.meshgrid(np.arange(len(lat_idx)), np.arange(len(lon_idx)), indexing="ij")
        flat_lat, flat_lon = glat.ravel(), glon.ravel()
        flat_iy, flat_ix = giy.ravel(), gix.ravel()

        n = min(self.max_regional_points, flat_lat.size)
        pick = rng.choice(flat_lat.size, size=n, replace=False)

        bbox = (
            float(self.lat[lat_idx[0]]),
            float(self.lat[lat_idx[-1]]),
            float(self.lon[lon_idx[0]]),
            float(self.lon[lon_idx[-1]]),
        )
        return lat_idx, lon_idx, flat_iy[pick], flat_ix[pick], flat_lat[pick], flat_lon[pick], bbox

    def _extract_points(self, t, lat_idx, lon_idx, iy, ix):
        """Stack standardised variables at sampled points for timestep t."""
        cols = []
        for v in self.variables:
            arr = self.data[v].isel(time=t, latitude=lat_idx, longitude=lon_idx).values
            col = (arr[iy, ix] - self.mean[v]) / self.std[v]
            cols.append(col)
        feat = np.stack(cols, axis=-1).astype(np.float32)
        return np.nan_to_num(feat, nan=0.0)

    def _extract_scalar(self, t, lat_i, lon_i):
        """Extract standardised variables at a single grid point for timestep t."""
        vals = []
        for v in self.variables:
            raw = float(self.data[v].isel(time=t, latitude=lat_i, longitude=lon_i).values)
            vals.append((raw - self.mean[v]) / self.std[v])
        arr = np.array(vals, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0)

    def _global_observations(self, t, bbox):
        """One IFS observation per coarse cell not refined away by the bbox.

        Returns (features_array [N_global, F], lat_lons list of tuples).
        """
        mesh_set = set(
            build_variable_resolution_mesh(bbox, self.coarse_res, self.fine_res)
        )
        feats, lls = [], []
        for cell in self._coarse_cells:
            if cell not in mesh_set:
                # This coarse cell was refined away — skip it.
                continue
            lat_i, lon_i = self._cell_grid_idx[cell]
            feats.append(self._extract_scalar(t, lat_i, lon_i))
            clat, clon = h3.cell_to_latlng(cell)
            lls.append((float(clat), float(clon)))
        return np.stack(feats, axis=0), lls

    def __getitem__(self, idx):
        """Return (features, lat_lons, target, bbox) for one sample.

        features/target are [N_global + N_regional, F]; lat_lons is a list of
        (lat, lon) tuples; bbox is (lat_min, lat_max, lon_min, lon_max).
        """
        rng = np.random.default_rng(self.seed + idx)
        lat_idx, lon_idx, iy, ix, plat, plon, bbox = self._sample_box(rng)

        reg_feat = self._extract_points(idx, lat_idx, lon_idx, iy, ix)
        reg_target = self._extract_points(idx + 1, lat_idx, lon_idx, iy, ix)
        reg_lls = [(float(a), float(b)) for a, b in zip(plat, plon)]

        glob_feat, glob_lls = self._global_observations(idx, bbox)
        glob_target_feat, _ = self._global_observations(idx + 1, bbox)

        features = torch.from_numpy(np.concatenate([glob_feat, reg_feat], axis=0))
        target = torch.from_numpy(np.concatenate([glob_target_feat, reg_target], axis=0))
        lat_lons = glob_lls + reg_lls

        return features, lat_lons, target, bbox
