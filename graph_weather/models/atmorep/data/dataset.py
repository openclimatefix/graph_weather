import os
import numpy as np
from torch.utils.data import Dataset

try:
    import xarray as xr
except ImportError:
    xr = None


class ERA5Dataset(Dataset):
    """
    ERA5Dataset for handling atmospheric data from the ARCO-ERA5 dataset.
    
    This implementation provides direct access to ERA5 data without requiring
    an index file, making it more general-purpose and easier to use.
    """
    
    def __init__(
        self,
        data_dir: str,
        fields: list = None,
        years: list = None,
        months: list = None,
        transform=None,
        file_pattern: str = "*.nc",
        use_dask: bool = False
    ):
        """
        Args:
            data_dir (str): Directory containing the ERA5 data files or GCP bucket path.
            fields (list, optional): List of fields/variables to load.
            years (list, optional): List of years to consider. Defaults to all available.
            months (list, optional): List of months to consider. Defaults to all months (1-12).
            transform (callable, optional): Optional transform to apply on a sample.
            file_pattern (str, optional): Pattern to match data files. Defaults to "*.nc".
            use_dask (bool, optional): Whether to use dask for lazy loading. Defaults to False.
        """
        self.data_dir = data_dir
        self.fields = fields if fields is not None else []
        self.years = years
        self.months = months
        self.transform = transform
        self.file_pattern = file_pattern
        self.use_dask = use_dask
        
        # Check if xarray is installed
        if xr is None:
            raise ImportError("xarray is required for ERA5Dataset. Install with: pip install xarray")
        
        # Build file list from directory or GCP bucket
        self.file_list = self._build_file_list()
        
        if not self.file_list:
            raise ValueError(f"No data files found in {data_dir} with pattern {file_pattern}")
    
    def _build_file_list(self):
        """
        Build a list of data files based on the provided criteria.
        Handles both local paths and GCP bucket paths.
        """
        file_list = []
        
        # Handle GCP bucket path
        if self.data_dir.startswith("gs://"):
            try:
                from google.cloud import storage
                
                # Parse bucket name and prefix
                bucket_name = self.data_dir.split("/")[2]
                prefix = "/".join(self.data_dir.split("/")[3:])
                
                # Initialize GCP client and get bucket
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                
                # List blobs in bucket with prefix
                blobs = bucket.list_blobs(prefix=prefix)
                
                # Filter by file pattern, years, and months if specified
                for blob in blobs:
                    filename = blob.name.split("/")[-1]
                    if self._matches_criteria(filename):
                        file_list.append(f"gs://{bucket_name}/{blob.name}")
                
            except ImportError:
                raise ImportError("google-cloud-storage is required for GCP access. "
                                 "Install with: pip install google-cloud-storage")
        
        # Handle local paths
        else:
            if not os.path.exists(self.data_dir):
                raise ValueError(f"Data directory {self.data_dir} does not exist")
            
            for root, _, files in os.walk(self.data_dir):
                for filename in files:
                    if self._matches_criteria(filename):
                        file_list.append(os.path.join(root, filename))
        
        return sorted(file_list)
    
    def _matches_criteria(self, filename):
        """
        Check if a filename matches the specified criteria (pattern, years, months).
        """
        # Check file pattern
        import fnmatch
        if not fnmatch.fnmatch(filename, self.file_pattern):
            return False
        
        # Additional filtering logic for years and months could be added here
        # This would depend on the specific filename format of the ERA5 data
        
        return True
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Retrieve a sample by loading data from the file at the given index.
        """
        file_path = self.file_list[idx]
        data = self.load_file(file_path)
        
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def load_file(self, file_path):
        """
        Load data from a netCDF file using xarray.
        
        Returns:
            dict: Dictionary containing the requested fields and their values.
        """
        # Load data using xarray
        if self.use_dask:
            ds = xr.open_dataset(file_path, engine="netcdf4", chunks={})
        else:
            ds = xr.open_dataset(file_path, engine="netcdf4")
        
        # Select only the requested fields if specified
        if self.fields:
            ds = ds[self.fields]
        
        # Convert to dictionary format
        data_dict = {}
        for var_name in ds.variables:
            if var_name not in ds.dims:  # Skip dimension variables
                data_dict[var_name] = ds[var_name].values
        
        # Close the dataset to free resources
        ds.close()
        
        return data_dict