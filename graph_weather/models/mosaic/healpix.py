"""HEALPix helpers for MOSAIC block-sparse attention.

With NESTED indexing a HEALPix pixel p subdivides into the four children
4p, 4p+1, 4p+2 and 4p+3, so contiguous index ranges correspond to spatial
neighbourhoods on the sphere (arXiv:2604.16429, Section 3.2). That is the
property block-sparse attention relies on: a block of consecutive tokens
is a compact region rather than an arbitrary set of points.

healpy is an optional dependency. It is listed in the pixi environment but
not in the package requirements, so it is imported lazily and only the
functions in this module need it. Block-sparse attention itself never
requires healpy: any locality-preserving ordering can be supplied by the
caller.
"""

from typing import Optional

import torch

try:  # pragma: no cover - exercised only when healpy is absent
    import healpy
except ImportError:
    healpy = None


def _check_nside(nside: int) -> None:
    """Validate that nside is a positive power of two, as NESTED requires."""
    if nside < 1 or (nside & (nside - 1)) != 0:
        raise ValueError(f"nside must be a positive power of two, got {nside}")


def _require_healpy() -> None:
    """Raise a helpful error when healpy is needed but not installed."""
    if healpy is None:
        raise ImportError(
            "healpy is required for HEALPix ordering. Install it with "
            "`pip install healpy`, or pass your own locality-preserving "
            "token order to BlockSparseAttention."
        )


def healpix_grid(nside: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the (lat, lon) coordinates of a HEALPix grid in NESTED order.

    Args:
        nside: HEALPix resolution parameter; the grid has 12 * nside ** 2
            pixels.
        dtype: Floating point dtype of the returned tensor.

    Returns:
        Tensor of shape (12 * nside ** 2, 2) holding (lat, lon) in radians.
    """
    _require_healpy()
    _check_nside(nside)
    n_pixels = 12 * nside * nside
    lon_deg, lat_deg = healpy.pix2ang(nside, torch.arange(n_pixels).numpy(), nest=True, lonlat=True)
    lat = torch.deg2rad(torch.as_tensor(lat_deg, dtype=dtype))
    lon = torch.deg2rad(torch.as_tensor(lon_deg, dtype=dtype))
    return torch.stack([lat, lon], dim=-1)


def nested_order(coords: torch.Tensor, nside: Optional[int] = None) -> torch.Tensor:
    """Order arbitrary points so contiguous blocks are spatial neighbours.

    Points are binned into HEALPix pixels in NESTED ordering and sorted by
    pixel index, which places nearby points next to each other regardless of
    the original ordering. This lets block-sparse attention run on point
    sets such as station networks, not only on regular grids.

    Args:
        coords: Tensor of shape (n_points, 2) with (lat, lon) in radians.
        nside: HEALPix resolution used for binning. Defaults to the smallest
            power of two that gives at least as many pixels as points.

    Returns:
        Long tensor of shape (n_points,) with the permutation that sorts the
        points into NESTED order.
    """
    _require_healpy()
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape (n_points, 2), got {tuple(coords.shape)}")
    n_points = coords.shape[0]
    if nside is None:
        nside = 1
        while 12 * nside * nside < n_points:
            nside *= 2
    else:
        _check_nside(nside)
    lat_deg = torch.rad2deg(coords[:, 0]).double().numpy()
    lon_deg = torch.rad2deg(coords[:, 1]).double().numpy()
    pixels = healpy.ang2pix(nside, lon_deg, lat_deg, nest=True, lonlat=True)
    return torch.argsort(torch.as_tensor(pixels, dtype=torch.long), stable=True)
