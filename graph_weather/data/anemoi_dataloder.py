from anemoi.datasets import open_dataset

def load_anemoi_dataset(dataset_name, region=None, time_range=None, **kwargs):
    """
    Load and return dataset using Anemoi.

    Args:
        dataset_name (str): Single dataset name like "gfs-2020-10" 
        region (dict): Optional. {'lat_min': .., 'lat_max': .., 'lon_min': .., 'lon_max': ..}
        time_range (tuple): Optional. ('YYYY-MM-DD', 'YYYY-MM-DD')
        **kwargs: Additional arguments passed to open_dataset

    Returns:
        Anemoi dataset object
    """
    # Build the configuration for opening the dataset
    config = {"dataset": dataset_name}
    
    # Add time range if provided
    if time_range:
        config["start"] = time_range[0]
        config["end"] = time_range[1]
    
    # Add any additional kwargs
    config.update(kwargs)
    
    # Open the dataset
    dataset = open_dataset(config)
    
    # Note: Regional subsetting would need to be handled differently
    # as Anemoi datasets may not support direct lat/lon slicing
    # This depends on the specific dataset structure
    
    return dataset

def load_multiple_datasets(dataset_names, **kwargs):
    """
    Load multiple datasets and combine them.
    
    Args:
        dataset_names (list): List of dataset names
        **kwargs: Arguments passed to individual dataset loading
    
    Returns:
        Combined Anemoi dataset
    """
    # For multiple datasets, use ensemble approach
    config = {
        "dataset": {
            "ensemble": dataset_names
        }
    }
    
    # Add other configuration
    if "time_range" in kwargs:
        config["start"] = kwargs["time_range"][0]
        config["end"] = kwargs["time_range"][1]
    
    return open_dataset(config)
