"""Dataset that samples movable high-resolution sub-grids from IFS analysis.

Each sample is a bounding box at a random location and timestep, drawn from the
regular lat/lon IFS ``hres_analysis`` grid. It yields a ``(features, lat_lons,
target, global_context)`` tuple shaped for ``RegionalForecaster.forward``: a
region that can move anywhere on the globe per sample (Issue #3).
"""

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

CORE_SURFACE = [
    "2_metre_temperature",
    "2_metre_dewpoint_temperature",
    "10_metre_u_wind_component",
    "10_metre_v_wind_component",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_cloud_cover",
    "total_column_water_vapour",
    "skin_temperature",
]

# Per-variable mean/std measured from 40 random IFS timesteps (stride-20 global
# subsample); see scripts/exploration/compute_ifs_stats.py. Override via mean/std args.
CORE_SURFACE_MEAN = {
    "2_metre_temperature": 278.9795,
    "2_metre_dewpoint_temperature": 274.3301,
    "10_metre_u_wind_component": -0.0102,
    "10_metre_v_wind_component": 0.1637,
    "mean_sea_level_pressure": 100925.1357,
    "surface_pressure": 96529.5451,
    "total_cloud_cover": 0.6729,
    "total_column_water_vapour": 19.0496,
    "skin_temperature": 279.3837,
}
CORE_SURFACE_STD = {
    "2_metre_temperature": 21.3684,
    "2_metre_dewpoint_temperature": 20.7277,
    "10_metre_u_wind_component": 5.647,
    "10_metre_v_wind_component": 4.8797,
    "mean_sea_level_pressure": 1351.9402,
    "surface_pressure": 9757.2853,
    "total_cloud_cover": 0.3823,
    "total_column_water_vapour": 16.9003,
    "skin_temperature": 22.4583,
}

DEFAULT_STORE = "bkr/ifs/hres_analysis.icechunk"


def open_ifs_store(store_url: str) -> xr.Dataset:
    """Open an IFS Icechunk store on Source Cooperative for anonymous reading."""
    import os

    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    import icechunk

    bucket, prefix = store_url.split("/", 1)
    storage = icechunk.s3_storage(
        bucket=bucket,
        prefix=prefix,
        endpoint_url="https://data.source.coop",
        region="us-east-1",
        anonymous=True,
        force_path_style=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False, zarr_format=3)


class RegionalDataset(Dataset):
    """Movable sub-grid samples from a regular lat/lon IFS analysis grid.

    Args:
        dataset: An open ``xarray.Dataset`` to sample from. When ``None``, the
            store at ``store_url`` is opened.
        store_url: Source Cooperative ``<bucket>/<prefix>`` of the IFS store.
        variables: Surface variable names to stack into the feature dimension.
        extent_deg: Side length of the square bounding box, in degrees.
        max_points: Cap on observation points per sample (subsampled if exceeded).
        seed: Base seed; sample ``idx`` uses ``seed + idx`` so boxes move per idx.
        mean: Per-variable means for standardisation.
        std: Per-variable standard deviations for standardisation.
        global_coarsen: Block-average factor for the coarse global context.

    Note:
        ``global_context`` is a coarse (block-averaged) low-resolution view of the
        same IFS field at the same points, returned for the BoundaryNudgingLayer.
        It is a Phase-1 stand-in: it does NOT yet inject true outside-domain
        weather, which needs a separate host model or a unified mesh (later).
    """

    def __init__(
        self,
        dataset: xr.Dataset = None,
        store_url: str = DEFAULT_STORE,
        variables: list = None,
        extent_deg: float = 20.0,
        max_points: int = 2000,
        seed: int = 0,
        mean: dict = None,
        std: dict = None,
        global_coarsen: int = 8,
    ):
        """Open the source grid and record sampling configuration."""
        super().__init__()
        self.data = dataset if dataset is not None else open_ifs_store(store_url)
        self.variables = variables if variables is not None else CORE_SURFACE
        self.extent_deg = extent_deg
        self.max_points = max_points
        self.seed = seed
        self.mean = mean if mean is not None else CORE_SURFACE_MEAN
        self.std = std if std is not None else CORE_SURFACE_STD
        self.global_coarsen = global_coarsen
        self.lat = self.data["latitude"].values
        self.lon = self.data["longitude"].values

    def __len__(self) -> int:
        """Number of t -> t+1 sample pairs."""
        return int(self.data.sizes["time"]) - 1

    def _sample_box(self, rng):
        """Pick a movable bbox center and return point grid indices and coords."""
        half = self.extent_deg / 2.0
        lat_c = rng.uniform(self.lat.min() + half, self.lat.max() - half)
        lon_c = rng.uniform(self.lon.min() + half, self.lon.max() - half)

        lat_idx = np.flatnonzero(np.abs(self.lat - lat_c) <= half)
        lon_idx = np.flatnonzero(np.abs(self.lon - lon_c) <= half)

        glat, glon = np.meshgrid(self.lat[lat_idx], self.lon[lon_idx], indexing="ij")
        giy, gix = np.meshgrid(np.arange(len(lat_idx)), np.arange(len(lon_idx)), indexing="ij")
        flat_lat, flat_lon = glat.ravel(), glon.ravel()
        flat_iy, flat_ix = giy.ravel(), gix.ravel()

        n = min(self.max_points, flat_lat.size)
        pick = rng.choice(flat_lat.size, size=n, replace=False)
        return (
            lat_idx,
            lon_idx,
            flat_iy[pick],
            flat_ix[pick],
            flat_lat[pick],
            flat_lon[pick],
        )

    def _coarsen(self, arr):
        """Block-average a 2D crop into kxk blocks, broadcast back to its shape."""
        k = self.global_coarsen
        if k <= 1:
            return arr
        ny, nx = arr.shape
        out = np.empty_like(arr)
        for by in range(0, ny, k):
            for bx in range(0, nx, k):
                block = arr[by : by + k, bx : bx + k]
                out[by : by + k, bx : bx + k] = (
                    np.nanmean(block) if np.isfinite(block).any() else np.nan
                )
        return out

    def _extract(self, t, lat_idx, lon_idx, iy, ix, coarse=False):
        """Stack standardised variables at the sampled points for timestep t.

        When ``coarse`` is set, each variable's crop is block-averaged first, so
        the sampled values form a low-resolution (global-context) view.
        """
        cols = []
        for v in self.variables:
            arr = self.data[v].isel(time=t, latitude=lat_idx, longitude=lon_idx).values
            if coarse:
                arr = self._coarsen(arr)
            col = (arr[iy, ix] - self.mean[v]) / self.std[v]
            cols.append(col)
        feat = np.stack(cols, axis=-1).astype(np.float32)
        return np.nan_to_num(feat, nan=0.0)

    def __getitem__(self, idx):
        """Return (features, lat_lons, target, global_context) for one box.

        features/target/global_context are [N, F]; lat_lons is a list of
        (lat, lon) tuples. global_context is the coarse view at the same points.
        """
        rng = np.random.default_rng(self.seed + idx)
        lat_idx, lon_idx, iy, ix, plat, plon = self._sample_box(rng)

        features = torch.from_numpy(self._extract(idx, lat_idx, lon_idx, iy, ix))
        target = torch.from_numpy(self._extract(idx + 1, lat_idx, lon_idx, iy, ix))
        global_context = torch.from_numpy(self._extract(idx, lat_idx, lon_idx, iy, ix, coarse=True))
        lat_lons = [(float(a), float(b)) for a, b in zip(plat, plon)]
        return features, lat_lons, target, global_context
