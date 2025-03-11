from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union

@dataclass(frozen=True)  
class AtmoRepConfig:
    """
    Configuration class for the Atmospheric Representation Model (AtmoRep). 
    This class holds the various configurations related to data, model dimensions, 
    training, and hierarchical sampling.

    Attributes:
        input_fields (List[str]): List of fields to be used as input to the model (e.g., 't2m', 'u10').
        spatial_dims (Tuple[int, int]): The spatial dimensions of the input data (latitude, longitude grid size).
        patch_size (int): Size of the spatial patches used for the vision transformer model.
        time_steps (int): Number of time steps to consider in training.
        mask_ratio (float): The ratio of tokens to mask during training to encourage robustness.
        model_dims (Dict[str, int]): The model's dimensionality settings, including encoder, decoder, etc.
        hidden_dim (int): The hidden dimension size used in the MultiFormer model.
        num_heads (int): Number of attention heads in the transformer architecture.
        num_layers (int): Number of layers in the transformer model.
        mlp_ratio (int): Ratio for the MLP hidden layer size.
        dropout (float): Dropout rate for the model to prevent overfitting.
        attention_dropout (float): Dropout rate specifically for attention layers.
        batch_size (int): Batch size used during training.
        learning_rate (float): Learning rate for the optimizer during training.
        weight_decay (float): Weight decay parameter for regularization.
        epochs (int): The number of epochs to train the model.
        warmup_epochs (int): Number of warmup epochs to gradually increase the learning rate.
        year_month_samples (int): Number of year-month pairs to sample per batch.
        time_slices_per_ym (int): Number of time slices to consider per year-month pair.
        neighborhoods_per_slice (Tuple[int, int]): Min and max neighborhoods per time slice.
        neighborhood_size (Tuple[int, int]): The spatial size of a neighborhood.
        num_ensemble_members (int): Number of ensemble members to use for final predictions.
    """

    # Data config
    input_fields: List[str] = field(default_factory=lambda: ['t2m', 'u10', 'v10', 'z500', 'msl'])
    spatial_dims: Tuple[int, int] = (128, 256)  # latitude, longitude grid size
    patch_size: int = 16  # spatial patch size for vision transformer
    time_steps: int = 24  # number of time steps to consider
    mask_ratio: float = 0.75  # ratio of tokens to mask during training
    
    # Model dimensions config
    model_dims: Dict[str, int] = field(default_factory=lambda: {
        'encoder': 768,
        'decoder': 768,
        'projection': 256,
        'embedding': 512
    })
    
    # MultiFormer config
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: int = 4
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training config
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 100
    warmup_epochs: int = 10
    
    # Hierarchical sampling config
    year_month_samples: int = 4  # number of year-month pairs to sample per batch
    time_slices_per_ym: int = 6  # number of time slices per year-month pair
    neighborhoods_per_slice: Tuple[int, int] = (2, 8)  # min and max neighborhoods per time slice
    neighborhood_size: Tuple[int, int] = (32, 32)  # spatial size of a neighborhood
    
    # Ensemble prediction config
    num_ensemble_members: int = 5
    
    def __post_init__(self):
        """
            Post-initialization method to validate configuration parameters after object creation.
            Ensures that fields like `spatial_dims`, `batch_size`, `learning_rate`, and `input_fields` are valid.
        """      
        # Validate spatial_dims: must be a tuple of two positive integers
        if not (isinstance(self.spatial_dims, tuple) and len(self.spatial_dims) == 2 and 
                all(isinstance(dim, int) and dim > 0 for dim in self.spatial_dims)):
            raise ValueError("spatial_dims must be a tuple of two positive integers")
        
        # Validate batch_size: must be a positive integer
        if not (isinstance(self.batch_size, int) and self.batch_size > 0):
            raise ValueError("batch_size must be a positive integer")
        
        # Validate learning_rate: must be a positive number
        if not (isinstance(self.learning_rate, (int, float)) and self.learning_rate > 0):
            raise ValueError("learning_rate must be a positive number")
        
        # Validate input_fields: must be a non-empty list
        if not (isinstance(self.input_fields, list) and len(self.input_fields) > 0):
            raise ValueError("input_fields must be a non-empty list")

    def __hash__(self):
        """Make the class hashable by creating a hash from its fields."""
        # Convert mutable attributes to immutable types for hashing
        input_fields_tuple = tuple(self.input_fields)
        model_dims_tuple = tuple((k, v) for k, v in sorted(self.model_dims.items()))
        
        # Create a tuple of all attributes and hash it
        return hash((
            input_fields_tuple,
            self.spatial_dims,
            self.patch_size,
            self.time_steps,
            self.mask_ratio,
            model_dims_tuple,
            self.hidden_dim,
            self.num_heads,
            self.num_layers,
            self.mlp_ratio,
            self.dropout,
            self.attention_dropout,
            self.batch_size,
            self.learning_rate,
            self.weight_decay,
            self.epochs,
            self.warmup_epochs,
            self.year_month_samples,
            self.time_slices_per_ym,
            self.neighborhoods_per_slice,
            self.neighborhood_size,
            self.num_ensemble_members
        ))