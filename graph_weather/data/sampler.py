import logging
import os
import random

import torch
import xarray as xr


class HierarchicalSampler:
    def __init__(self, config: dict, data_path: str, loader=None):
        """
        Generic Hierarchical Data Sampler.

        Args:
            config (dict): Configuration parameters, e.g.:
                - input_fields: List[str], fields to load.
                - time_segment_samples: int, number of time segments to sample.
                - time_slices_per_segment: int, number of time slices per segment.
                - time_steps: int, number of consecutive time steps to extract.
                - neighborhoods_per_slice: tuple(int, int), range for the number of spatial patches.
                - neighborhood_size: tuple(int, int), patch size in (latitude, longitude).
                - patch_size: int, size for tokenization patches.
                - mask_ratio: float, fraction of tokens to mask.
                - years: tuple(int, int), (start_year, end_year); used for default time segments.
                - file_pattern: str, pattern to locate files (default: "{field}_{year}_{month:02d}.nc").
            data_path (str): Base directory containing the data files.
            loader (callable, optional): Custom function for loading a dataset. Should accept a time_segment and field,
                and return an xarray.Dataset (or None if not found). If not provided, a default loader is used.
        """
        self.config = config
        self.data_path = data_path
        self.loader = loader if loader is not None else self._default_loader

        # Create a list of available time segments (e.g., year-month pairs)
        self.time_segments = self._get_available_time_segments()
        self.logger = logging.getLogger("HierarchicalSampler")
        self.logger.setLevel(logging.INFO)

    def _get_available_time_segments(self):
        """
        Generate a list of available time segments based on the provided years.
        Default implementation assumes data is organized by year and month.

        Returns:
            list of tuples: Each tuple is (year, month).
        """
        start_year, end_year = self.config.get("years", (1979, 2022))
        years = list(range(start_year, end_year + 1))
        months = list(range(1, 13))
        return [(year, month) for year in years for month in months]

    def _default_loader(self, time_segment, field):
        """
        Default loader function to load a dataset for a given time segment and field.
        Expects data files to follow a naming pattern.

        Args:
            time_segment (tuple): Typically (year, month).
            field (str): The field name to load.

        Returns:
            xarray.Dataset or None
        """
        year, month = time_segment
        file_pattern = self.config.get("file_pattern", "{field}_{year}_{month:02d}.nc")
        file_path = os.path.join(
            self.data_path, file_pattern.format(field=field, year=year, month=month)
        )
        try:
            if os.path.exists(file_path):
                ds = xr.open_dataset(file_path)
                return ds
            else:
                self.logger.warning(f"File not found: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading data for {time_segment}, field {field}: {str(e)}")
            return None

    def sample_batch(self, fields: list = None):
        """
        Samples a batch of data and corresponding masks from random time segments.

        Args:
            fields (list, optional): List of fields to load. Defaults to config["input_fields"].

        Returns:
            tuple: (batch_data, batch_masks) as dictionaries mapping each field to a tensor batch.
        """
        if fields is None:
            fields = self.config.get("input_fields", [])

        num_samples = self.config.get("time_segment_samples", 1)
        segments = random.sample(self.time_segments, num_samples)

        batch_data = {field: [] for field in fields}
        batch_masks = {field: [] for field in fields}
        time_steps = self.config.get("time_steps", 1)

        for segment in segments:
            for _ in range(self.config.get("time_slices_per_segment", 1)):
                for field in fields:
                    ds = self.loader(segment, field)
                    if ds is None or "time" not in ds.dims:
                        continue

                    # Sample a contiguous block of time
                    if len(ds.time) < time_steps:
                        continue
                    start_time = random.randint(0, len(ds.time) - time_steps)
                    time_slice = slice(start_time, start_time + time_steps)

                    # Determine spatial patch size and number of neighborhoods to sample
                    neighborhood_min, neighborhood_max = self.config.get(
                        "neighborhoods_per_slice", (1, 1)
                    )
                    num_neighborhoods = random.randint(neighborhood_min, neighborhood_max)
                    lat_size, lon_size = self.config.get("neighborhood_size", (1, 1))
                    patch_size = self.config.get("patch_size", 1)
                    mask_ratio = self.config.get("mask_ratio", 0.0)

                    for _ in range(num_neighborhoods):
                        # Check if dataset has the required spatial dimensions
                        if "latitude" not in ds.dims or "longitude" not in ds.dims:
                            self.logger.warning(
                                f"Dataset for field {field} missing latitude/longitude dims."
                            )
                            continue

                        ds_lat = ds["latitude"]
                        ds_lon = ds["longitude"]
                        if ds_lat.size < lat_size or ds_lon.size < lon_size:
                            self.logger.warning(
                                f"Spatial dimensions too small for field {field} in segment {segment}."
                            )
                            continue

                        lat_start = random.randint(0, ds_lat.size - lat_size)
                        lon_start = random.randint(0, ds_lon.size - lon_size)

                        chunk = (
                            ds[field]
                            .isel(
                                time=time_slice,
                                latitude=slice(lat_start, lat_start + lat_size),
                                longitude=slice(lon_start, lon_start + lon_size),
                            )
                            .values
                        )

                        # Build a mask for tokenized representation
                        h_patches = lat_size // patch_size
                        w_patches = lon_size // patch_size
                        mask_flat = torch.zeros(time_steps, h_patches * w_patches)
                        num_tokens = mask_flat.numel()
                        num_mask = int(num_tokens * mask_ratio)
                        if num_tokens > 0 and num_mask > 0:
                            mask_indices = random.sample(range(num_tokens), num_mask)
                            for idx in mask_indices:
                                t, p = divmod(idx, mask_flat.shape[1])
                                mask_flat[t, p] = 1

                        batch_data[field].append(torch.tensor(chunk, dtype=torch.float32))
                        batch_masks[field].append(mask_flat)

        # Stack samples for each field
        for field in fields:
            if batch_data[field]:
                batch_data[field] = torch.stack(batch_data[field])
                batch_masks[field] = torch.stack(batch_masks[field])
            else:
                default_shape = (0, time_steps, *self.config.get("neighborhood_size", (0, 0)))
                default_mask_shape = (
                    0,
                    time_steps,
                    (
                        self.config.get("neighborhood_size", (0, 0))[0]
                        // self.config.get("patch_size", 1)
                    )
                    * (
                        self.config.get("neighborhood_size", (0, 0))[1]
                        // self.config.get("patch_size", 1)
                    ),
                )
                batch_data[field] = torch.empty(default_shape)
                batch_masks[field] = torch.empty(default_mask_shape)

        return batch_data, batch_masks
