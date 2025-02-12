import torch

class Pressure3dConvNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: int, padding: int):
        super(Pressure3dConvNet, self).__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Surface2dConvNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: int, padding: int):
        super(Surface2dConvNet, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

