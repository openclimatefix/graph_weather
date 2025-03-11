import os
import random
import logging
import torch
import xarray as xr
from pathlib import Path
import numpy as np
from ..config import AtmoRepConfig

class FieldNormalizer:
    def __init__(self, config, stats_dir, create_stats=False):
        """
        Args:
            config (AtmoRepConfig): Configuration containing input_fields.
            stats_dir (Path): Directory containing per-field statistics files.
            create_stats (bool): If True, calculate statistics; otherwise, load from disk.
        """
        self.config = config
        self.stats_dir = stats_dir
        self.stats = {}
        if create_stats:
            self.calculate_stats()
        else:
            for field in config.input_fields:
                stats_file = stats_dir / f"{field}_stats.npy"
                if stats_file.exists():
                    loaded = np.load(stats_file, allow_pickle=True).item()
                    self.stats[field] = loaded
                else:
                    raise Exception(f"Stats file for {field} not found")
                    
    def calculate_stats(self):
        """
        Placeholder implementation to calculate statistics.
        In practice, you would compute the mean and std from your dataset.
        
        Returns:
            dict: A dictionary of statistics for each field.
        """
        self.stats = {field: {'mean': 0.0, 'std': 1.0} for field in self.config.input_fields}
        # Optionally, save the computed stats to disk:
        for field, stat in self.stats.items():
            np.save(self.stats_dir / f"{field}_stats.npy", stat)
        return self.stats
    
    def normalize(self, data, field):
        """
        Normalize the data for a given field.
        
        Args:
            data (torch.Tensor or np.array): Data to be normalized.
            field (str): Field name.
            
        Returns:
            Normalized data.
        """
        if field not in self.config.input_fields:
            raise Exception(f"Field {field} is not in the configuration")
        stats = self.stats.get(field, None)
        if stats is None:
            raise Exception(f"Stats for field {field} not available")
        mean = stats['mean']
        std = stats['std']
        return (data - mean) / std
    
    def denormalize(self, data, field):
        """
        Denormalize the data for a given field.
        
        Args:
            data (torch.Tensor or np.array): Data to be denormalized.
            field (str): Field name.
            
        Returns:
            Denormalized data.
        """
        if field not in self.config.input_fields:
            raise Exception(f"Field {field} is not in the configuration")
        stats = self.stats.get(field, None)
        if stats is None:
            raise Exception(f"Stats for field {field} not available")
        mean = stats['mean']
        std = stats['std']
        return data * std + mean
