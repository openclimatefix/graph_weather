import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)

    def __call__(self, x):
        # Downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Upsampling part
        upconv3 = self.upconv3(conv3)
        if upconv3.shape[-2:] != conv2.shape[-2:]:
            upconv3 = F.interpolate(upconv3, size=conv2.shape[-2:], mode='nearest')

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return expand


class ThermalizerLayer(nn.Module):
    def __init__(self, input_dim: int = 256, timesteps: int = 1000):
        super().__init__()
        self.score_model = UNet(input_dim, input_dim)
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

    def forward(self, x: torch.Tensor, t: torch.Tensor,height: int, width: int, batch: int) -> torch.Tensor:
        """
        Applies the Thermalizer denoising step.

        Args:
            x: The input tensor. Shape: (batch * nodes, features)
            t: The timestep (can be an integer scalar)

        Returns:
            The denoised tensor. Shape: same as input (batch * nodes, features)
        """
    # The input is (batch * nodes, features), but UNet needs (batch, features, height, width)
    # So we assume a square spatial shape (height x width = nodes)
    # This reshaping is required for compatibility with the UNet architecture
        batch_nodes, features = x.shape
        nodes = height * width
        assert batch * height * width == batch_nodes, "Height x Width x Batch must equal total nodes"

        # Reshape x to match UNet's input format
        x_reshaped = x.reshape(batch, height, width, features).permute(0, 3, 1, 2)  # (1, features, H, W)

        # Add Gaussian noise based on the current timestep
        noise = torch.randn_like(x_reshaped)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].to(x.device)
        sqrt_one_minus_alphas_cumprod_t = (1. - self.alphas_cumprod[t]).sqrt().to(x.device)
        noisy_x = sqrt_alphas_cumprod_t * x_reshaped + sqrt_one_minus_alphas_cumprod_t * noise

        # Predict the noise using the score model (UNet)
        score = self.score_model(noisy_x)

        # Estimate the original input from the noisy input and predicted noise
        pred_noise = (noisy_x - sqrt_alphas_cumprod_t * score) / sqrt_one_minus_alphas_cumprod_t

        # Reshape the output back to original input shape: (batch * nodes, features)
        return pred_noise.permute(0, 2, 3, 1).reshape(batch_nodes, features)
