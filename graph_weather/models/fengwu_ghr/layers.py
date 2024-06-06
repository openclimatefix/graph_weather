import numpy as np
from scipy.interpolate import griddata, interpn
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def grid_interpolate(lat_lons: list, z: torch.Tensor,
                     height, width,
                     method: str = "cubic"):
    # TODO 1. CPU only
    #      2. The mesh is a rectangle, not a sphere

    xi = np.arange(0.5, width, 1)/width*360
    yi = np.arange(0.5, height, 1)/height*180

    xi, yi = np.meshgrid(xi, yi)
    z = rearrange(z, "b n c -> n b c")
    z = griddata(
        lat_lons, z, (xi, yi),
        fill_value=0, method=method)
    z = rearrange(z, "h w b c -> b c h w")  # hw ?
    z = torch.tensor(z)
    return z

def grid_extrapolate(lat_lons, z,
                     height, width,
                     method: str = "cubic"):
    xi = np.arange(0.5, width, 1)/width*360
    yi = np.arange(0.5, height, 1)/height*180
    z = rearrange(z, "b c h w -> h w b c")
    z = z.detach().numpy()
    z= interpn((xi,yi),z, lat_lons,
               bounds_error=False,
                method=method)
    z = rearrange(z, "n b c -> b n c")
    z = torch.tensor(z)
    return z


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [Attention(dim, heads=heads, dim_head=dim_head),
                     FeedForward(dim, mlp_dim)]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class MetaModel(nn.Module):
    def __init__(self, lat_lons: list, *,
                 patch_size, depth,
                 heads, mlp_dim,
                 resolution=(721, 1440),
                 channels=3, dim_head=64,
                 interp_method='cubic'):
        super().__init__()
        image_height, image_width = pair(resolution)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        # interpolate
        self.interpolate = lambda z: grid_interpolate(
            lat_lons, z, image_height, image_width,
            method=interp_method)

        patch_dim = channels * patch_height * patch_width
        dim = patch_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p_h) (w p_w) -> b (h w) (p_h p_w c)",
                p_h=patch_height, p_w=patch_width
            ),
            nn.LayerNorm(patch_dim),  # TODO Do we need this?
            nn.Linear(patch_dim, dim),  # TODO Do we need this?
            nn.LayerNorm(dim),  # TODO Do we need this?
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.reshaper = nn.Sequential(
            Rearrange(
                "b (h w) (p_h p_w c) -> b c (h p_h) (w p_w)",
                h=image_height // patch_height,
                w=image_width // patch_width,
                p_h=patch_height,
                p_w=patch_width,
            )
        )

        # extrapolate
        self.extrapolate = lambda z: grid_extrapolate(
            lat_lons, z, image_height, image_width,
            method=interp_method)


    def forward(self, x):
        device = x.device
        dtype = x.dtype

        x = self.interpolate(x.to("cpu"))
        x = x.to(device, dtype=dtype)

        x = self.to_patch_embedding(x)
        x += self.pos_embedding.to(device, dtype=dtype)

        x = self.transformer(x)

        x = self.reshaper(x)

        x = self.extrapolate(x.to("cpu"))
        x = x.to(device, dtype=dtype)

        return x
