# Self-Supervised Data Assimilation Framework with 3D-Var Loss

## Overview

This implementation provides a complete self-supervised data assimilation framework that learns to produce analysis states by minimizing the 3D-Var cost function without using ground-truth labels. The system consists of neural networks that take background states and observations as input and produce optimal analysis states.

## Core Components

### 1. 3D-Var Loss Function (`data_assimilation.py`)

The `ThreeDVarLoss` class implements the core 3D-Var objective function:

```
J(x) = (x - x_b)^T B^{-1} (x - x_b) + (y - Hx)^T R^{-1} (y - Hx)
```

Where:
- `x`: analysis state (model output)
- `x_b`: background state (first guess)
- `y`: observations
- `B`: background error covariance
- `R`: observation error covariance
- `H`: observation operator

Key features:
- Supports custom background and observation error covariances
- Handles different observation operators
- Works with both fully connected and convolutional models
- Self-supervised (no ground-truth required)

### 2. Data Assimilation Models (`data_assimilation.py`)

Two model architectures are provided:

#### DataAssimilationModel
- Fully connected neural network
- Takes concatenated background and observations as input
- Produces analysis state as output
- Configurable hidden dimensions and layers

#### SimpleDataAssimilationModel
- Convolutional neural network for spatial data
- Works with 1D/2D grid data
- Preserves spatial relationships
- Efficient for gridded meteorological data

### 3. Data Pipeline (`assimilation_dataloader.py`)

- `AssimilationDataset`: Dataset class for background/observation pairs
- `AssimilationDataModule`: PyTorch Lightning-style data module
- Synthetic data generation with spatial correlation
- Observation masking and operator creation
- Train/validation/test splitting

### 4. Training Framework (`training_loop.py`)

- `DataAssimilationTrainer`: Complete training loop with validation
- Self-supervised training using 3D-Var loss
- Learning rate scheduling
- Model checkpointing
- Multi-mode training (good/poor background, sparse observations)

### 5. Evaluation Metrics (`evaluation.py`)

Comprehensive evaluation including:
- RMSE, MAE, bias calculations
- Correlation coefficients
- Spatial metrics
- Information gain
- Baseline comparisons
- Cross-validation

### 6. Visualization Tools (`visualization.py`)

- Training curves plotting
- Comparison grids (background, observations, analysis, true state)
- Error maps visualization
- RMSE comparisons
- Heatmaps and scatter plots
- Comprehensive dashboard

## Key Features

### Self-Supervised Learning
- No ground-truth labels required
- Physics-based loss function
- Learns optimal combination of background and observations

### Flexible Architecture
- Works with different grid sizes
- Supports multiple channels/variables
- Configurable network depth and width
- Multiple activation functions

### Multiple Training Modes
- With good first guess (low background error)
- With poor first guess (cold start)
- With varying observation densities
- Different error covariance specifications

### Comprehensive Evaluation
- Comparison with classical baselines
- Improvement metrics
- Spatial analysis
- Statistical validation

## Usage Example

```python
from graph_weather.graph_weather.models.data_assimilation import SimpleDataAssimilationModel, ThreeDVarLoss
from graph_weather.graph_weather.data.assimilation_dataloader import AssimilationDataModule
from graph_weather.graph_weather.models.training_loop import train_data_assimilation_model

# Create data module
data_module = AssimilationDataModule(
    grid_size=(16, 16),
    num_channels=1,
    bg_error_std=0.5,
    obs_error_std=0.3,
    obs_fraction=0.6
)
data_module.setup()

# Initialize model
model = SimpleDataAssimilationModel(
    grid_size=(16, 16),
    num_channels=1,
    hidden_dim=64,
    num_layers=3
)

# Train model
trainer, results = train_data_assimilation_model(
    model=model,
    train_loader=data_module.train_dataloader(),
    val_loader=data_module.val_dataloader(),
    epochs=100,
    lr=1e-3
)
```

## Mathematical Foundation

The 3D-Var cost function is based on Bayesian estimation theory:

```
J(x) = (x - x_b)^T B^{-1} (x - x_b) + (y - Hx)^T R^{-1} (y - Hx)
```

Where the first term represents the background constraint and the second term represents the observation constraint. The neural network learns to find the optimal balance between these constraints without explicit supervision.

## Advantages Over Classical Methods

1. **Learned Error Covariances**: The neural network can learn complex, non-linear relationships
2. **Adaptive Combination**: Automatically adjusts weighting based on data quality
3. **Scalability**: Can handle high-dimensional state spaces efficiently
4. **End-to-End Learning**: Optimizes the complete assimilation process

## Applications

This framework is suitable for:
- Weather forecasting data assimilation
- Climate model state estimation
- Oceanographic data assimilation
- Any physical system with background models and observations