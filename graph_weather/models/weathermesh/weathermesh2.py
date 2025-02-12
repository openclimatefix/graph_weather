from graph_weather.models.weathermesh.encoders import Pressure3dConvNet, Surface2dConvNet
import torch

"""
Notes on implementation

To make NATTEN work on a sphere, we implement our own circular padding. At the poles, we use the bump attention behavior from NATTEN. For position encoding of tokens, we use Rotary Embeddings.

In the default configuration of WeatherMesh 2, the NATTEN window is 5,7,7 in depth, width, height, corresponding to a physical size of 14 degrees longitude and latitude. WeatherMesh 2 contains two processors: a 6hr and a 1hr processor. Each is 10 NATTEN layers deep.

Training: distributed shampoo: https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md

Fork version of pytorch checkpoint library called matepoint to implement offloading to RAM

"""

class WeatherMesh(torch.nn.Module):
    def __init__(self):
        super(WeatherMesh, self).__init__()
        self.pressure_encoder = Pressure3dConvNet(1, 64, (3, 3, 3), 1, 1)
        self.surface_encoder = Surface2dConvNet(1, 64, (3, 3), 1, 1)

    def forward(self, pressure: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
        pressure = self.pressure_encoder(pressure)
        surface = self.surface_encoder(surface)
        # TODO Add the transformers layers here as the processor
        # TODO Add the decoders here as the output, which are inverse of the encoders
        return pressure, surface