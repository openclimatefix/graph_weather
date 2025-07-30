import torch
from torch import nn
import torch.nn.functional as F
import math


class AdaptiveUNet(nn.Module):
    """UNet that adapts its architecture based on input size"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Standard UNet components
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)
        
        # Simple network for small inputs
        self.simple_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, x):
        original_size = x.shape[-2:]
        
        # Use simple network for very small inputs
        if min(original_size) <= 4:
            return self.simple_net(x)
        
        # Use standard UNet for larger inputs
        return self._forward_standard(x)
    
    def _forward_standard(self, x):
        """Standard UNet forward pass"""
        original_size = x.shape[-2:]
        
        # Downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Upsampling part
        upconv3 = self.upconv3(conv3)
        
        # Ensure upconv3 matches conv2 size for concatenation
        if upconv3.shape[-2:] != conv2.shape[-2:]:
            upconv3 = F.interpolate(upconv3, size=conv2.shape[-2:], mode='bilinear', align_corners=False)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        
        # Ensure upconv2 matches conv1 size for concatenation
        if upconv2.shape[-2:] != conv1.shape[-2:]:
            upconv2 = F.interpolate(upconv2, size=conv1.shape[-2:], mode='bilinear', align_corners=False)
            
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        
        # Ensure final output matches original input size
        if upconv1.shape[-2:] != original_size:
            upconv1 = F.interpolate(upconv1, size=original_size, mode='bilinear', align_corners=False)

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        # Use GroupNorm instead of BatchNorm to avoid issues with small batches/spatial dims
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return expand


class ThermalizerLayer(nn.Module):
    def __init__(self, input_dim: int = 256, timesteps: int = 1000):
        super().__init__()
        self.score_model = AdaptiveUNet(input_dim, input_dim)
        self.timesteps = timesteps
        self.betas = self._cosine_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _infer_grid_dimensions(self, total_nodes):
        """
        Infer grid dimensions from total number of nodes.
        Prefer dimensions that work well with neural networks.
        """
        # For very small grids, just use the factorization
        if total_nodes <= 16:
            sqrt_nodes = int(math.sqrt(total_nodes))
            if sqrt_nodes * sqrt_nodes == total_nodes:
                return sqrt_nodes, sqrt_nodes
            
            # Find best factorization for small grids
            for h in range(1, total_nodes + 1):
                if total_nodes % h == 0:
                    w = total_nodes // h
                    if abs(h - w) <= 2:  # Prefer roughly square
                        return h, w
            return 1, total_nodes
        
        # For larger grids, try to find dimensions that work well
        sqrt_nodes = int(math.sqrt(total_nodes))
        
        # Try perfect square first
        if sqrt_nodes * sqrt_nodes == total_nodes:
            return sqrt_nodes, sqrt_nodes
        
        # Find factorization closest to square
        best_diff = float('inf')
        best_h, best_w = 1, total_nodes
        
        for h in range(max(1, sqrt_nodes - 5), sqrt_nodes + 6):
            if total_nodes % h == 0:
                w = total_nodes // h
                diff = abs(h - w)
                if diff < best_diff:
                    best_diff = diff
                    best_h, best_w = h, w
        
        return best_h, best_w

    def forward(self, x: torch.Tensor, t: torch.Tensor, height: int = None, width: int = None, batch: int = None) -> torch.Tensor:
        """
        Applies the Thermalizer denoising step.

        Args:
            x: The input tensor. Shape: (batch * nodes, features) or (total_nodes, features)
            t: The timestep (can be an integer scalar)
            height: Optional height dimension (will be inferred if not provided)
            width: Optional width dimension (will be inferred if not provided)  
            batch: Optional batch size (will be inferred if not provided)

        Returns:
            The denoised tensor. Shape: same as input
        """
        total_nodes, features = x.shape
        
        # If dimensions not provided, try to infer them
        if height is None or width is None or batch is None:
            if batch is None:
                # Assume batch size of 1 if not specified
                batch = 1
                nodes = total_nodes
            else:
                nodes = total_nodes // batch
            
            if height is None or width is None:
                height, width = self._infer_grid_dimensions(nodes)
        
        nodes = height * width
        
        # Validate dimensions
        if batch * nodes != total_nodes:
            raise ValueError(f"Dimension mismatch: batch({batch}) * height({height}) * width({width}) = {batch * nodes} != total_nodes({total_nodes})")

        # Reshape x to match UNet's input format
        x_reshaped = x.reshape(batch, height, width, features).permute(0, 3, 1, 2)  # (batch, features, H, W)

        # Add Gaussian noise based on the current timestep
        noise = torch.randn_like(x_reshaped)
        
        # Handle scalar timestep
        if isinstance(t, int):
            t = torch.tensor(t, device=x.device)
        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.item()
        
        # Ensure t is within valid range
        t = max(0, min(t, self.timesteps - 1))
        
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().to(x.device)
        sqrt_one_minus_alphas_cumprod_t = (1. - self.alphas_cumprod[t]).sqrt().to(x.device)
        
        noisy_x = sqrt_alphas_cumprod_t * x_reshaped + sqrt_one_minus_alphas_cumprod_t * noise

        # Predict the noise using the score model (UNet)
        score = self.score_model(noisy_x)

        # Estimate the original input from the noisy input and predicted noise
        pred_x = (noisy_x - sqrt_one_minus_alphas_cumprod_t * score) / sqrt_alphas_cumprod_t

        # Reshape the output back to original input shape: (batch * nodes, features)
        return pred_x.permute(0, 2, 3, 1).reshape(total_nodes, features)