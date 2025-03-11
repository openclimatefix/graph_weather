import os
import random
import logging
import torch
import xarray as xr
from ..config import AtmoRepConfig

# Hierarchical Data Sampler for efficient loading
class HierarchicalERA5Sampler:
    def __init__(self, config: AtmoRepConfig, era5_path: str):
        self.config = config
        self.era5_path = era5_path
        
        # Get available year-month pairs
        self.year_month_pairs = self._get_available_year_month_pairs()
        
        # Setup logging
        self.logger = logging.getLogger("HierarchicalSampler")
        self.logger.setLevel(logging.INFO)
    
    def _get_available_year_month_pairs(self):
        # This would be implemented to scan the ERA5 directory and find available data
        # For demonstration, we'll create a dummy list
        years = list(range(1979, 2023))
        months = list(range(1, 13))
        return [(year, month) for year in years for month in months]
    
    def _load_data_chunk(self, year: int, month: int, field: str):
        # In a real implementation, this would load the ERA5 data for a specific year-month
        # For demonstration, we'll create a dummy dataset
        try:
            file_path = os.path.join(self.era5_path, f"{field}_{year}_{month:02d}.nc")
            if os.path.exists(file_path):
                ds = xr.open_dataset(file_path)
                return ds
            else:
                self.logger.warning(f"File not found: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading data for {year}-{month}: {str(e)}")
            return None
    
    def sample_batch(self, fields: list = None):
        if fields is None:
            fields = self.config.input_fields
        
        # Sample year-month pairs
        ym_pairs = random.sample(self.year_month_pairs, self.config.year_month_samples)
        
        batch_data = {field: [] for field in fields}
        batch_masks = {field: [] for field in fields}
        
        for year, month in ym_pairs:
            # Sample time slices for this year-month
            for _ in range(self.config.time_slices_per_ym):
                field_data = {}
                
                # Load data for each field
                for field in fields:
                    ds = self._load_data_chunk(year, month, field)
                    if ds is None:
                        continue
                    
                    # Sample a random starting time
                    time_len = len(ds.time)
                    if time_len < self.config.time_steps:
                        continue
                    
                    start_time = random.randint(0, time_len - self.config.time_steps)
                    time_slice = slice(start_time, start_time + self.config.time_steps)
                    
                    # Sample spatial neighborhoods
                    num_neighborhoods = random.randint(
                        self.config.neighborhoods_per_slice[0],
                        self.config.neighborhoods_per_slice[1]
                    )
                    
                    for _ in range(num_neighborhoods):
                        # Sample random spatial location
                        lat_size, lon_size = self.config.neighborhood_size
                        lat_start = random.randint(0, ds.latitude.size - lat_size)
                        lon_start = random.randint(0, ds.longitude.size - lon_size)
                        
                        # Extract data chunk
                        chunk = ds[field].isel(
                            time=time_slice,
                            latitude=slice(lat_start, lat_start + lat_size),
                            longitude=slice(lon_start, lon_start + lon_size)
                        ).values
                        
                        # Create mask for this chunk
                        mask = torch.zeros((self.config.time_steps, lat_size, lon_size))
                        mask_flat = torch.zeros(self.config.time_steps, (lat_size // self.config.patch_size) * (lon_size // self.config.patch_size))
                        
                        # Randomly mask tokens
                        num_tokens = mask_flat.numel()
                        num_mask = int(num_tokens * self.config.mask_ratio)
                        mask_indices = random.sample(range(num_tokens), num_mask)
                        for idx in mask_indices:
                            t, p = divmod(idx, mask_flat.shape[1])
                            mask_flat[t, p] = 1
                        
                        # Append to batch
                        batch_data[field].append(torch.tensor(chunk, dtype=torch.float32))
                        
                        # Reshape mask to match the tokenized representation
                        h_patches = lat_size // self.config.patch_size
                        w_patches = lon_size // self.config.patch_size
                        batch_masks[field].append(mask_flat)
        
        # Stack the collected samples
        for field in fields:
            if batch_data[field]:
                batch_data[field] = torch.stack(batch_data[field])
                batch_masks[field] = torch.stack(batch_masks[field])
            else:
                # No data for this field
                batch_data[field] = torch.zeros((0, self.config.time_steps, *self.config.neighborhood_size))
                batch_masks[field] = torch.zeros((0, self.config.time_steps, (self.config.neighborhood_size[0] // self.config.patch_size) * (self.config.neighborhood_size[1] // self.config.patch_size)))
        
        return batch_data, batch_masks