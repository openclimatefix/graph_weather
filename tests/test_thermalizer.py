import torch
from graph_weather.models.layers.thermalizer import ThermalizerLayer

def test_thermalizer_forward_shape():
    batch, nodes, features = 2, 144, 256
    height, width = 12, 12  # 10x10 = 100 nodes
    assert height * width == nodes

    x = torch.randn(batch, features, height, width)  # shape (2, 256, 10, 10)

    # Flatten spatial dimensions back to match ThermalizerLayer interface
    x_flat = x.permute(0, 2, 3, 1).reshape(batch * nodes, features)
    x_flat = x_flat.reshape(batch * nodes, features)  # still (288, 256)

    layer = ThermalizerLayer(input_dim=features)
    t = torch.randint(0, 1000, (1,)).item()

    out = layer(x_flat, t, height=height, width=width, batch=batch)
    assert out.shape == x_flat.shape
