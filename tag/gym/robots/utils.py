import numpy as np


def tile_xyz(n, z, spacing=1.0, device=None, dtype=None):
    side = int(np.ceil(np.sqrt(n)))

    # Compute grid indices
    idx = np.arange(n)
    row = idx // side
    col = idx % side

    offset = spacing * (side - 1) / 2.0
    x = (row * spacing) - offset
    y = (col * spacing) - offset
    z = np.full_like(x, z)

    return np.stack([x, y, z], axis=1)  # shape (n, 3)
