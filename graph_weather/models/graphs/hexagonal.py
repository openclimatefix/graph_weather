"""Generate hexagonal global grid using Uber's H3 library."""
import h3
import numpy as np


def generate_hexagonal_grid(resolution: int = 2) -> np.ndarray:
    """Generate hexagonal global grid using Uber's H3 library.

    Args:
        resolution: H3 resolution level

    Returns:
        Hexagonal grid
    """
    base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
    base_h3_map = {h_i: i for i, h_i in enumerate(base_h3_grid)}
    return np.array(base_h3_grid), base_h3_map
