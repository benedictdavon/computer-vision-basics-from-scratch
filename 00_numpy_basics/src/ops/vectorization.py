from __future__ import annotations

import numpy as np


def mean_per_image_loop(images: np.ndarray) -> np.ndarray:
    """Compute per-image mean using an explicit Python loop.

    Args:
        images: (B, H, W, C)

    Returns:
        means: (B,)
    """
    B = images.shape[0]
    means = np.zeros((B,), dtype=np.float32)
    for i in range(B):
        means[i] = float(np.mean(images[i]))
    return means


def mean_per_image_vectorized(images: np.ndarray) -> np.ndarray:
    """Compute per-image mean using vectorized NumPy ops.

    Args:
        images: (B, H, W, C)

    Returns:
        means: (B,)
    """
    return images.mean(axis=(1, 2, 3)).astype(np.float32)
