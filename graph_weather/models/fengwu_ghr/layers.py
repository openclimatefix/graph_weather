"""Building blocks for the FengWu-GHR model.

This module gathers the layers used to run a vision-transformer backbone over weather data:
interpolation between an irregular set of lat/lon points and a regular grid, fixed sine-cosine
positional embeddings, the attention and feed-forward blocks, the `ImageMetaModel` backbone and
its lat/lon counterpart `MetaModel`, the wrappers that run either of them at a higher
resolution by splitting the input into sub-images, and LoRA layers for fine-tuning.
"""

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch_geometric.nn.pool import knn
from torch_geometric.utils import scatter


def pair(t):
    """
    Return `t` as a two-element tuple.

    Args:
        t: A tuple, which is returned unchanged, or any other value, which is duplicated.

    Returns:
        `t` itself if it is already a tuple, otherwise `(t, t)`.
    """
    return t if isinstance(t, tuple) else (t, t)


def knn_interpolate(
    x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor, k: int = 4, num_workers: int = 1
):
    """
    Interpolate features from one set of positions onto another.

    Every target position takes the weighted mean of the features of its `k` nearest source
    positions, weighted by the inverse of the squared distance between them.

    Args:
        x: Features of the source nodes, of shape `(num_source_nodes, num_features)`.
        pos_x: Coordinates of the source nodes, of shape `(num_source_nodes, num_dims)`.
        pos_y: Coordinates of the target nodes, of shape `(num_target_nodes, num_dims)`.
        k: Number of nearest source nodes used for each target node.
        num_workers: Number of workers used by the k-nearest-neighbor search.

    Returns:
        The interpolated features, of shape `(num_target_nodes, num_features)`.
    """
    with torch.no_grad():
        assign_index = knn(pos_x, pos_y, k, num_workers=num_workers)
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

        y_idx, x_idx = y_idx.to(x.device), x_idx.to(x.device)
        weights = weights.to(x.device)

    den = scatter(weights, y_idx, 0, pos_y.size(0), reduce="sum")
    y = scatter(x[x_idx] * weights, y_idx, 0, pos_y.size(0), reduce="sum")

    y = y / den

    return y


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    """
    Build a fixed two-dimensional sine-cosine positional embedding.

    Args:
        h: Number of positions along the height of the grid.
        w: Number of positions along the width of the grid.
        dim: Size of the embedding. Must be a multiple of 4.
        temperature: Base of the geometric progression used for the frequencies.
        dtype: Data type of the returned embedding.

    Returns:
        The positional embedding, of shape `(h * w, dim)`.

    Raises:
        AssertionError: If `dim` is not a multiple of 4.
    """
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
    """Pre-norm position-wise feed-forward block."""

    def __init__(self, dim, hidden_dim):
        """
        Initialize FeedForward.

        Args:
            dim: Size of the input and of the output features.
            hidden_dim: Size of the hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        """
        Apply the feed-forward network.

        Args:
            x: Input tensor whose last dimension is `dim`.

        Returns:
            A tensor with the same shape as `x`.
        """
        return self.net(x)


class Attention(nn.Module):
    """Pre-norm multi-head self-attention block."""

    def __init__(self, dim, heads=8, dim_head=64):
        """
        Initialize Attention.

        Args:
            dim: Size of the input and of the output features.
            heads: Number of attention heads.
            dim_head: Size of each attention head.
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        """
        Apply multi-head self-attention along the sequence dimension.

        Args:
            x: Input tensor of shape `(batch, sequence, dim)`.

        Returns:
            A tensor with the same shape as `x`.
        """
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Stack of residual attention and feed-forward blocks, followed by a layer norm.

    When `res` is True the batch is assumed to hold the `s_h * s_w` sub-images of a single
    image, and each block is followed by an extra attention layer applied across those
    sub-images: the sequence is regrouped so that attention runs over the `s_h * s_w` axis for
    each patch position, and is then restored to its original layout.
    """

    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim, res=False, image_size=None, scale_factor=None
    ):
        """
        Initialize Transformer.

        Args:
            dim: Size of the input and of the output features.
            depth: Number of attention and feed-forward blocks.
            heads: Number of attention heads.
            dim_head: Size of each attention head.
            mlp_dim: Size of the hidden layer of the feed-forward blocks.
            res: Whether to add the extra attention layers acting across sub-images.
            image_size: Size `(h, w)` in patches of one sub-image, or a single int if it is
                square. Only used when `res` is True.
            scale_factor: Number `(s_h, s_w)` of sub-images along each axis, or a single int
                for the same value on both axes. Only used when `res` is True.

        Raises:
            AssertionError: If `res` is True and `image_size` or `scale_factor` is None.
        """
        super().__init__()
        self.depth = depth
        self.res = res
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(
                nn.ModuleList(
                    [Attention(dim, heads=heads, dim_head=dim_head), FeedForward(dim, mlp_dim)]
                )
            )
            if self.res:
                assert (
                    image_size is not None and scale_factor is not None
                ), "If res=True, you must provide h, w and scale_factor"
                h, w = pair(image_size)
                s_h, s_w = pair(scale_factor)
                self.res_layers.append(
                    nn.ModuleList(
                        [  # reshape to original shape     window partition operation
                            #  (b s_h s_w) (h w) d -> b (s_h h) (s_w w) d  -> (b h w) (s_h s_w) d
                            Rearrange(
                                "(b s_h s_w) (h w) d -> (b h w) (s_h s_w) d",
                                h=h,
                                w=w,
                                s_h=s_h,
                                s_w=s_w,
                            ),
                            # TODO ?????
                            Attention(dim, heads=heads, dim_head=dim_head),
                            # restore shape
                            Rearrange(
                                "(b h w) (s_h s_w) d -> (b s_h s_w) (h w) d",
                                h=h,
                                w=w,
                                s_h=s_h,
                                s_w=s_w,
                            ),
                        ]
                    )
                )

    def forward(self, x):
        """
        Run every block over the input sequence.

        Args:
            x: Input tensor of shape `(batch, sequence, dim)`.

        Returns:
            The normalized output tensor, of the same shape as `x`.
        """
        for i in range(self.depth):
            attn, ff = self.layers[i]
            x = attn(x) + x
            x = ff(x) + x
            if self.res:
                reshape, loc_attn, restore = self.res_layers[i]
                x = reshape(x)
                x = loc_attn(x) + x
                x = restore(x)
        return self.norm(x)


class ImageMetaModel(nn.Module):
    """
    Vision-transformer backbone mapping an image to an image of the same shape.

    The input is cut into non-overlapping patches, each patch is embedded, a fixed sine-cosine
    positional embedding is added, a `Transformer` is applied and the patches are folded back
    into an image. The embedding size is the flattened size of one patch, so the number of
    channels is preserved.
    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        depth,
        heads,
        mlp_dim,
        channels,
        dim_head,
        res=False,
        scale_factor=None,
        **kwargs,
    ):
        """
        Initialize ImageMetaModel.

        Args:
            image_size: Size `(height, width)` of the image, or a single int if it is square.
            patch_size: Size `(height, width)` of a patch, or a single int if it is square.
            depth: Number of transformer blocks.
            heads: Number of attention heads.
            mlp_dim: Size of the hidden layer of the feed-forward blocks.
            channels: Number of channels of the image.
            dim_head: Size of each attention head.
            res: Whether the transformer gets the extra attention layers across sub-images.
            scale_factor: Number `(s_h, s_w)` of sub-images along each axis, or a single int
                for the same value on both axes.
            **kwargs: Ignored. Lets the attributes of an existing `ImageMetaModel` be passed
                straight through when a rescaled copy of it is built.

        Raises:
            AssertionError: If `res` is True while `scale_factor` is None, or if the image size
                is not divisible by the patch size.
        """
        super().__init__()
        # TODO this can probably be done better
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.channels = channels
        self.dim_head = dim_head
        self.res = res
        self.scale_factor = scale_factor

        self.image_height, self.image_width = pair(image_size)
        self.patch_height, self.patch_width = pair(patch_size)
        s_h, s_w = pair(scale_factor)

        if res:
            assert scale_factor is not None, "If res=True, you must provide scale_factor"
        assert (
            self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * self.patch_height * self.patch_width
        dim = patch_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p_h) (w p_w) -> b (h w) (p_h p_w c)",
                p_h=self.patch_height,
                p_w=self.patch_width,
            ),
            nn.LayerNorm(patch_dim),  # TODO Do we need this?
            nn.Linear(patch_dim, dim),  # TODO Do we need this?
            nn.LayerNorm(dim),  # TODO Do we need this?
        )

        self.pos_embedding = posemb_sincos_2d(
            h=self.image_height // self.patch_height,
            w=self.image_width // self.patch_width,
            dim=dim,
        )

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            res=res,
            image_size=(
                self.image_height // self.patch_height,
                self.image_width // self.patch_width,
            ),
            scale_factor=(s_h, s_w),
        )

        self.reshaper = nn.Sequential(
            Rearrange(
                "b (h w) (p_h p_w c) -> b c (h p_h) (w p_w)",
                h=self.image_height // self.patch_height,
                w=self.image_width // self.patch_width,
                p_h=self.patch_height,
                p_w=self.patch_width,
            )
        )

    def forward(self, x):
        """
        Embed the image into patches, run the transformer and rebuild the image.

        Args:
            x: Input tensor of shape `(batch, channels, height, width)`.

        Returns:
            A tensor with the same shape as `x`.

        Raises:
            AssertionError: If `x` does not have `channels` channels.
        """
        assert x.shape[1] == self.channels, "Wrong number of channels"
        device = x.device
        dtype = x.dtype

        x = self.to_patch_embedding(x)
        x += self.pos_embedding.to(device, dtype=dtype)

        x = self.transformer(x)
        x = self.reshaper(x)

        return x


class WrapperImageModel(nn.Module):
    """
    Run an `ImageMetaModel` on an image `scale_factor` times larger than its own.

    The image is cut into `s_h * s_w` sub-images that are stacked along the batch dimension, a
    copy of the given model built with `res=True` is applied to them, and the sub-images are
    then put back together. The weights of the given model are loaded into that copy.
    """

    def __init__(self, image_meta_model: ImageMetaModel, scale_factor):
        """
        Initialize WrapperImageModel.

        Args:
            image_meta_model: Model whose settings and weights are reused by the `res=True` copy.
            scale_factor: Number `(s_h, s_w)` of sub-images along each axis, or a single int
                for the same value on both axes.
        """
        super().__init__()
        s_h, s_w = pair(scale_factor)
        self.batcher = Rearrange("b c (h s_h) (w s_w) -> (b s_h s_w) c h w", s_h=s_h, s_w=s_w)

        imm_args = vars(image_meta_model)
        imm_args.update({"res": True, "scale_factor": scale_factor})
        self.image_meta_model = ImageMetaModel(**imm_args)
        self.image_meta_model.load_state_dict(image_meta_model.state_dict(), strict=False)

        self.debatcher = Rearrange("(b s_h s_w) c h w -> b c (h s_h) (w s_w)", s_h=s_h, s_w=s_w)

    def forward(self, x):
        """
        Split the image into sub-images, run the wrapped model and reassemble them.

        Args:
            x: Input tensor of shape `(batch, channels, height * s_h, width * s_w)`.

        Returns:
            A tensor with the same shape as `x`.
        """
        x = self.batcher(x)
        x = self.image_meta_model(x)
        x = self.debatcher(x)
        return x


class MetaModel(nn.Module):
    """
    Apply an `ImageMetaModel` to data defined on an irregular set of lat/lon points.

    The point values are interpolated onto a regular lat/lon grid, the image backbone is run on
    that grid, and the output is interpolated back onto the original points.
    """

    def __init__(
        self,
        lat_lons: list,
        *,
        image_size,
        patch_size,
        depth,
        heads,
        mlp_dim,
        channels,
        dim_head=64,
    ):
        """
        Initialize MetaModel.

        Args:
            lat_lons: List of `(lat, lon)` coordinates of the input points.
            image_size: Size `(height, width)` of the grid, or a single int if it is square.
            patch_size: Size `(height, width)` of a patch, or a single int if it is square.
            depth: Number of transformer blocks.
            heads: Number of attention heads.
            mlp_dim: Size of the hidden layer of the feed-forward blocks.
            channels: Number of channels of the gridded data.
            dim_head: Size of each attention head.
        """
        super().__init__()
        self.i_h, self.i_w = pair(image_size)

        self.pos_x = torch.tensor(lat_lons).to(torch.long)
        self.pos_y = torch.cartesian_prod(
            (torch.arange(-self.i_h / 2, self.i_h / 2, 1) / self.i_h * 180).to(torch.long),
            (torch.arange(0, self.i_w, 1) / self.i_w * 360).to(torch.long),
        )

        self.image_meta_model = ImageMetaModel(
            image_size=image_size,
            patch_size=patch_size,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dim_head=dim_head,
        )

    def forward(self, x):
        """
        Interpolate onto the grid, run the image model and interpolate back to the points.

        Args:
            x: Input tensor of shape `(batch, num_points, channels)`.

        Returns:
            A tensor with the same shape as `x`.
        """
        b, n, c = x.shape

        x = rearrange(x, "b n c -> n (b c)")
        x = knn_interpolate(x, self.pos_x, self.pos_y)
        x = rearrange(x, "(h w) (b c) -> b c h w", b=b, c=c, h=self.i_h, w=self.i_w)
        x = self.image_meta_model(x)

        x = rearrange(x, "b c h w -> (h w) (b c)")
        x = knn_interpolate(x, self.pos_y, self.pos_x)
        x = rearrange(x, "n (b c) -> b n c", b=b, c=c)
        return x


class WrapperMetaModel(nn.Module):
    """
    Run a `MetaModel` on a lat/lon grid `scale_factor` times finer than its own.

    The points are interpolated onto the finer grid, that grid is cut into `s_h * s_w`
    sub-images, a copy of the wrapped image backbone built with `res=True` is applied to them,
    and the result is reassembled and interpolated back onto the original points.
    """

    def __init__(self, lat_lons: list, meta_model: MetaModel, scale_factor):
        """
        Initialize WrapperMetaModel.

        Args:
            lat_lons: List of `(lat, lon)` coordinates of the input points.
            meta_model: Model whose grid size is scaled up and whose image backbone settings
                and weights are reused by the `res=True` copy.
            scale_factor: Number `(s_h, s_w)` of sub-images along each axis, or a single int
                for the same value on both axes.
        """
        super().__init__()
        s_h, s_w = pair(scale_factor)
        self.i_h, self.i_w = meta_model.i_h * s_h, meta_model.i_w * s_w
        self.pos_x = torch.tensor(lat_lons)
        self.pos_y = torch.cartesian_prod(
            (torch.arange(-self.i_h / 2, self.i_h / 2, 1) / self.i_h * 180).to(torch.long),
            (torch.arange(0, self.i_w, 1) / self.i_w * 360).to(torch.long),
        )

        self.batcher = Rearrange("b c (h s_h) (w s_w) -> (b s_h s_w) c h w", s_h=s_h, s_w=s_w)

        imm_args = vars(meta_model.image_meta_model)
        imm_args.update({"res": True, "scale_factor": scale_factor})
        self.image_meta_model = ImageMetaModel(**imm_args)
        self.image_meta_model.load_state_dict(
            meta_model.image_meta_model.state_dict(), strict=False
        )

        self.debatcher = Rearrange("(b s_h s_w) c h w -> b c (h s_h) (w s_w)", s_h=s_h, s_w=s_w)

    def forward(self, x):
        """
        Interpolate onto the finer grid, run the wrapped image model and interpolate back.

        Args:
            x: Input tensor of shape `(batch, num_points, channels)`.

        Returns:
            A tensor with the same shape as `x`.
        """
        b, n, c = x.shape

        x = rearrange(x, "b n c -> n (b c)")
        x = knn_interpolate(x, self.pos_x, self.pos_y)
        x = rearrange(x, "(h w) (b c) -> b c h w", b=b, c=c, h=self.i_h, w=self.i_w)

        x = self.batcher(x)
        x = self.image_meta_model(x)
        x = self.debatcher(x)

        x = rearrange(x, "b c h w -> (h w) (b c)")
        x = knn_interpolate(x, self.pos_y, self.pos_x)
        x = rearrange(x, "n (b c) -> b n c", b=b, c=c)

        return x


class LoRALayer(nn.Module):
    """Linear layer with an added low-rank term of rank `r`."""

    def __init__(self, linear_layer: nn.Module, r: int):
        """
        Initialize LoRALayer.

        Args:
            linear_layer (nn.Module): Linear layer to be transformed.
            r (int): rank of the low-rank matrix.
        """
        super().__init__()
        out_features, in_features = linear_layer.weight.shape

        self.A = nn.Parameter(torch.randn(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.linear_layer = linear_layer

    def forward(self, x):
        """
        Add the low-rank term to the output of the wrapped linear layer.

        Args:
            x: Input tensor.

        Returns:
            The sum of the linear layer output and of the low-rank term.
        """
        out = self.linear_layer(x) + self.B @ self.A @ x
        return out


class LoRAModule(nn.Module):
    """Model whose linear layers are replaced by `LoRALayer` and set to evaluation mode."""

    def __init__(self, model, r=4):
        """
        Initialize LoRAModule.

        Args:
            model (nn.Module): Model to be modified with LoRA layers.
            r (int, optional): Rank of LoRA layers. Defaults to 4.
        """
        super().__init__()
        for name, layer in model.named_modules():
            layer.eval()
            if isinstance(layer, nn.Linear):
                lora_layer = LoRALayer(layer, r)
                setattr(model, name, lora_layer)
        self.model = model

    def forward(self, x):
        """
        Run the wrapped model.

        Args:
            x: Input tensor.

        Returns:
            The output of the wrapped model.
        """
        return self.model(x)
