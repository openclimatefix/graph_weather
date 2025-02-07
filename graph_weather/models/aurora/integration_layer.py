import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from typing import Dict, Union, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
from torch_geometric.nn import knn
from torch_geometric.utils import scatter
from ..gencast.utils.noise import generate_isotropic_noise

# Import Fengwu_GHR specific components
from graph_weather.models.fengwu_ghr import (
    ImageMetaModel,
    WrapperImageModel,
)

class ModelType(Enum):
    FENGWU_GHR = "fengwu_ghr"
    GENCAST = "gencast"

@dataclass
class GenCastConfig:
    """Configuration for GenCast model."""
    hidden_dims: List[int] = None
    num_blocks: int = 3
    num_heads: int = 4
    splits: int = 0
    num_hops: int = 1

@dataclass
class Fengwu_GHRConfig:
    """Configuration for Fengwu_GHR model."""
    image_size: Union[int, Tuple[int, int]]
    patch_size: Union[int, Tuple[int, int]]
    depth: int
    heads: int
    mlp_dim: int
    channels: int
    dim_head: int = 64
    scale_factor: Optional[Union[int, Tuple[int, int]]] = None

class TransformationError(Exception):
    """Custom exception for transformation errors."""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class IntegrationLayer:
    """
    Enhanced integration layer for compatibility between Fengwu_GHR and GenCast implementations.
    Includes model configurations, advanced tensor transformations, and error handling.
    """
    
    def __init__(self, 
                 grid_lon: np.ndarray,
                 grid_lat: np.ndarray,
                 input_features_dim: int,
                 output_features_dim: int,
                 gencast_config: Optional[GenCastConfig] = None,
                 fengwu_ghr_config: Optional[Fengwu_GHRConfig] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize integration layer with model configurations.
        """
        self.device = device
        self.grid_lon = self._validate_grid(grid_lon, "longitude")
        self.grid_lat = self._validate_grid(grid_lat, "latitude")
        self.input_features_dim = input_features_dim
        self.output_features_dim = output_features_dim
        
        # Store model configurations
        self.gencast_config = gencast_config or GenCastConfig()
        self.fengwu_ghr_config = fengwu_ghr_config
        
        # Initialize Fengwu_GHR model if config is provided
        if self.fengwu_ghr_config:
            self.fengwu_model = self._initialize_fengwu_model()
        
        # Calculate dimensions
        self.num_lon = len(grid_lon)
        self.num_lat = len(grid_lat)
        
        # Initialize tensor transformation patterns
        self._init_transform_patterns()

    def _initialize_fengwu_model(self) -> Union[ImageMetaModel, WrapperImageModel]:
        """
        Initialize Fengwu_GHR model based on configuration.
        """
        if not self.fengwu_ghr_config:
            return None
            
        base_model = ImageMetaModel(
            image_size=self.fengwu_ghr_config.image_size,
            patch_size=self.fengwu_ghr_config.patch_size,
            depth=self.fengwu_ghr_config.depth,
            heads=self.fengwu_ghr_config.heads,
            mlp_dim=self.fengwu_ghr_config.mlp_dim,
            channels=self.fengwu_ghr_config.channels,
            dim_head=self.fengwu_ghr_config.dim_head
        )
        
        # If scale factor is provided, wrap the model
        if self.fengwu_ghr_config.scale_factor:
            return WrapperImageModel(
                base_model,
                scale_factor=self.fengwu_ghr_config.scale_factor
            )
            
        return base_model

    def _init_transform_patterns(self):
        """Initialize common tensor transformation patterns."""
        self.transform_patterns = {
            'b_c_h_w_to_b_n_c': lambda x: rearrange(x, 'b c h w -> b (h w) c'),
            'b_n_c_to_b_c_h_w': lambda x: rearrange(x, 'b (h w) c -> b c h w', 
                                                   h=self.num_lat, w=self.num_lon),
            'b_h_w_c_to_b_n_c': lambda x: rearrange(x, 'b h w c -> b (h w) c'),
            'b_n_c_to_b_h_w_c': lambda x: rearrange(x, 'b (h w) c -> b h w c',
                                                   h=self.num_lat, w=self.num_lon)
        }

    def _validate_grid(self, grid: np.ndarray, grid_type: str) -> np.ndarray:
        """
        Validate grid points.
        """
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)
            
        if grid_type == "longitude":
            if not (grid >= 0).all() or not (grid < 360).all():
                raise ValidationError(f"Longitude values must be between 0 and 360")
        elif grid_type == "latitude":
            if not (grid >= -90).all() or not (grid <= 90).all():
                raise ValidationError(f"Latitude values must be between -90 and 90")
                
        return grid

    def _validate_tensor(self, x: torch.Tensor, expected_shape: tuple = None) -> None:
        """
        Validate input tensor.
        """
        if not isinstance(x, torch.Tensor):
            raise ValidationError(f"Input must be a torch.Tensor, got {type(x)}")
            
        if expected_shape and x.shape != expected_shape:
            raise ValidationError(f"Expected shape {expected_shape}, got {x.shape}")
            
        if torch.isnan(x).any():
            raise ValidationError("Input tensor contains NaN values")

    def generate_noise(self, 
                      num_samples: int = 1,
                      isotropic: bool = True,
                      seed: Optional[int] = None) -> np.ndarray:
        """
        Generate noise compatible with GenCast's noise structure.
        """
        if seed is not None:
            np.random.seed(seed)
            
        try:
            noise = generate_isotropic_noise(
                num_lon=self.num_lon,
                num_lat=self.num_lat,
                num_samples=num_samples,
                isotropic=isotropic
            )
                
            return noise
            
        except Exception as e:
            raise TransformationError(f"Error generating noise: {str(e)}")

    def preprocess_coordinates(self, 
                             lat_lons: list,
                             validate: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process coordinates for both Fengwu_GHR and GenCast formats.
        """
        try:
            if validate:
                lat_lons = np.array(lat_lons)
                if lat_lons.shape[1] != 2:
                    raise ValidationError("lat_lons must be a list of [lat, lon] pairs")
                    
            pos_x = torch.tensor(lat_lons).to(torch.long)
            pos_y = torch.cartesian_prod(
                torch.tensor(self.grid_lat, dtype=torch.long),
                torch.tensor(self.grid_lon, dtype=torch.long)
            )
            
            return pos_x.to(self.device), pos_y.to(self.device)
        except Exception as e:
            raise TransformationError(f"Error processing coordinates: {str(e)}")

    def transform_tensor(self, 
                        x: torch.Tensor,
                        source_format: str,
                        target_format: str) -> torch.Tensor:
        """
        Apply tensor transformation using predefined patterns.
        """
        transform_key = f"{source_format}_to_{target_format}"
        if transform_key not in self.transform_patterns:
            raise TransformationError(f"Unsupported transformation: {transform_key}")
            
        try:
            return self.transform_patterns[transform_key](x)
        except Exception as e:
            raise TransformationError(f"Error during tensor transformation: {str(e)}")

    def interpolate_features(self,
                         x: torch.Tensor,
                         pos_x: torch.Tensor,
                         pos_y: torch.Tensor,
                         k: int = 4,
                         weighted: bool = True) -> torch.Tensor:
        """
        Perform KNN interpolation on features with additional options.
        """
        try:
            with torch.no_grad():
                # Find the nearest neighbors (k neighbors)
                assign_index = knn(pos_x, pos_y, k)
                y_idx, x_idx = assign_index[0], assign_index[1]

                if weighted:
                    # Compute squared distances for weighting
                    diff = pos_x[x_idx] - pos_y[y_idx]
                    squared_distance = (diff * diff).sum(dim=-1, keepdim=True)  # Squared distance
                    weights = 1.0 / torch.clamp(squared_distance, min=1e-16)  # Inverse distance
                else:
                    weights = torch.ones_like(y_idx, dtype=torch.float)

                # Move tensors to the same device as x
                y_idx = y_idx.to(x.device)
                x_idx = x_idx.to(x.device)
                weights = weights.to(x.device)

            # Adjust scatter operation to match tensor sizes properly
            den = scatter(weights, y_idx, 0, pos_y.size(0), reduce="sum")  # Denominator: sum of weights

            # Ensure that x[x_idx] * weights is correctly dimensioned to match pos_y size
            weighted_features = scatter(x[x_idx] * weights, y_idx, 0, pos_y.size(0), reduce="sum")  # Weighted features

            # Normalize the result by the denominator
            result = weighted_features / den

            return result

        except Exception as e:
            raise TransformationError(f"Error during feature interpolation: {str(e)}")
