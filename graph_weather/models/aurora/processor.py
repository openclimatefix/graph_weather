"""
    Perceiver Transformer Processor:
    - Takes encoded features and processes them using latent space mapping.
    - Uses a latent-space bottleneck to compress input dimensions.
    - Provides an efficient way to extract long-range dependencies.
"""
import torch
import torch.nn as nn
from transformers import PerceiverModel, PerceiverConfig

class PerceiverProcessor(nn.Module):
    def __init__(self, latent_dim=512, input_dim=96, max_seq_len=4096):
        """
        Args:
            latent_dim (int): Number of latent units in the Perceiver.
            input_dim (int): Dimension of the input features.
            max_seq_len (int): Maximum input sequence length.
        """
        super().__init__()

        # Perceiver configuration
        config = PerceiverConfig(
            hidden_size=input_dim,  # Feature dimension of input (this is typically set to d_model)
            num_latents=latent_dim,  # Number of latents in latent space
            latent_dim=latent_dim,   # Dimension of each latent
            max_position_embeddings=max_seq_len,  # Maximum sequence length
            d_model=1280             # d_model is the internal model dimension for processing
        )

        # Initialize the Perceiver model with the configuration
        self.perceiver = PerceiverModel(config)

        # Input projection to match the input_dim to d_model (1280)
        self.input_projection = nn.Linear(input_dim, 1280)

        # Output projection to reduce the dimensionality from d_model (1280) to latent_dim (512)
        self.projection = nn.Linear(1280, latent_dim)

    def forward(self, x):
        """
        Forward pass for the Perceiver Processor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            torch.Tensor: Latent representation processed by Perceiver.
        """
        batch_size = x.shape[0]

        # Handle case where x has 4 dimensions (e.g., (batch_size, seq_len, height, width))
        if len(x.shape) == 4:
            seq_len = x.shape[1] * x.shape[2] * x.shape[3]   
            x = x.view(batch_size, seq_len, -1)
        else:
            # If x has 3 dimensions (batch_size, seq_len, input_dim)
            seq_len = x.shape[1]  # Just use the sequence length
            x = x.view(batch_size, seq_len, -1)

        # Apply the input projection to match input_dim to d_model (1280)
        x = self.input_projection(x)

        # Process with Perceiver
        output = self.perceiver(inputs=x).last_hidden_state  # Access processed latents

        # Apply the output projection to reduce dimensionality to latent_dim (512)
        latent_output = self.projection(output.mean(dim=1))  # Reduce over the sequence length (dim=1)

        return latent_output