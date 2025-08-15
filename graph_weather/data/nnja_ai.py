"""
Dynamic loader for NNJA-AI datasets with support for primary descriptors and data variables.

Features:
- Automatically loads primary descriptors + primary data by default
- Supports custom variable selection
- Can load all variables when requested
- Returns xarray.Dataset with time as the only coordinate
- Optimized for performance with direct xarray access

"""

import numpy as np
import xarray as xr
from torch.utils.data import Dataset

try:
    from nnja import DataCatalog
except ImportError:
    raise ImportError(
        "NNJA-AI library not installed. Install with: "
        "`pip install git+https://github.com/brightbandtech/nnja-ai.git`"
    )


def _classify_variable(nnja_var) -> str:
    """Return category of a variable using attributes or repr fallback."""
    # First try to get explicit attributes
    if hasattr(nnja_var, "category"):
        return nnja_var.category
    if hasattr(nnja_var, "role"):
        return nnja_var.role

    # Fallback to string representation
    tag = repr(nnja_var).lower()
    if "primary_descriptor" in tag or "primary descriptor" in tag:
        return "primary_descriptor"
    if "primary_data" in tag or "primary data" in tag:
        return "primary_data"
    return "other"


def load_nnja_dataset(
    dataset_name: str,
    time=None,
    variables: list[str] | None = None,
    load_all: bool = False,
) -> xr.Dataset:
    """
    Load a NNJA dataset as an xarray.Dataset with time as the only coordinate.

    Args:
        dataset_name: Name of NNJA dataset to load
        time: Time selection (single timestamp, slice, or None)
        variables: Specific variables to load (overrides default)
        load_all: Load all available variables in the dataset

    Returns:
        xarray.Dataset with only 'time' dimension/coordinate
    """
    try:
        cat = DataCatalog()
        ds_meta = cat[dataset_name]
        ds_meta.load_manifest()
    except KeyError as e:
        raise ValueError(f"Dataset '{dataset_name}' not found in catalog") from e

    vars_dict = ds_meta.variables
    if load_all:
        vars_to_load = list(vars_dict.keys())
    elif variables:
        # Validate requested variables
        invalid_vars = [v for v in variables if v not in vars_dict]
        if invalid_vars:
            raise ValueError(f"Invalid variables requested: {invalid_vars}")
        vars_to_load = variables
    else:
        # Default: primary descriptors + primary data
        primary = [
            name
            for name, v in vars_dict.items()
            if _classify_variable(v) in ("primary_descriptor", "primary_data")
        ]
        vars_to_load = primary

    try:
        df = ds_meta.sel(time=time, variables=vars_to_load).load_dataset(
            backend="pandas", engine="pyarrow"
        )
    except Exception as e:
        raise RuntimeError(f"Error loading dataset '{dataset_name}': {str(e)}") from e

    xrds = df.to_xarray()

    # Standardize coordinate names
    rename_map = {"OBS_TIMESTAMP": "time", "LAT": "latitude", "LON": "longitude"}
    xrds = xrds.rename({k: v for k, v in rename_map.items() if k in xrds})

    # Ensure 'time' coordinate exists
    if "time" not in xrds and "OBS_DATE" in xrds:
        xrds = xrds.rename({"OBS_DATE": "time"})

    # Handle time conversion if needed
    if "time" in xrds and not np.issubdtype(xrds.time.dtype, np.datetime64):
        xrds["time"] = xrds.time.astype("datetime64[ns]")

    # If time is not a dimension but 'obs' is, swap
    if "time" in xrds and "obs" in xrds.dims and "time" not in xrds.dims:
        xrds = xrds.swap_dims({"obs": "time"})
        if "obs" in xrds.coords:
            xrds = xrds.reset_coords("obs", drop=True)

    if "time" in xrds and "time" not in xrds.coords:
        xrds = xrds.set_coords("time")

    # Flatten extra dimensions into time
    extra_dims = [d for d in xrds.dims if d != "time"]
    if extra_dims:
        time_values = xrds.time.values if "time" in xrds else None
        xrds = xrds.stack(sample=tuple(extra_dims))
        xrds = xrds.reset_index("sample")

        # Rename to time and restore original time values
        if "sample" in xrds.dims:
            xrds = xrds.swap_dims({"sample": "time"})
            if "sample" in xrds.coords:
                xrds = xrds.reset_coords("sample", drop=True)
        if time_values is not None:
            xrds["time"] = ("time", time_values)

    if "time" not in xrds.dims:
        raise RuntimeError("Failed to establish 'time' dimension in output dataset")

    return xrds


class SensorDataset(Dataset):
    """PyTorch Dataset wrapper for NNJA-AI datasets with optimized access."""

    def __init__(self, dataset_name, time=None, variables=None, load_all=False):
        """Initialize dataset loader.

        Args:
            dataset_name: Name of NNJA dataset to load
            time: Time selection (single timestamp or slice)
            variables: Specific variables to load
            load_all: If True, loads all available variables
        """
        self.dataset_name = dataset_name
        self.time = time

        self.xrds = load_nnja_dataset(
            dataset_name, time=time, variables=variables, load_all=load_all
        )

        # Store for efficient access
        self.variables = list(self.xrds.data_vars.keys())
        self.time_index = self.xrds.time.values

    def __len__(self):
        return self.xrds.sizes["time"]

    def __getitem__(self, idx):
        """Direct xarray access without DataFrame conversion."""
        time_point = self.time_index[idx]
        return {
            var: self.xrds[var].sel(time=time_point).item() for var in self.variables
        }


class NNJAXarrayAsTorchDataset(Dataset):
    """Adapter for torch Dataset directly from xarray."""

    def __init__(self, xrds):
        """Initialize adapter.

        Args:
            xrds: xarray Dataset to convert
        """
        self.ds = xrds
        self.vars = list(xrds.data_vars.keys())
        self.time_index = xrds.time.values

    def __len__(self):
        return self.ds.sizes["time"]

    def __getitem__(self, idx):
        time_point = self.time_index[idx]
        return {var: self.ds[var].sel(time=time_point).item() for var in self.vars}
