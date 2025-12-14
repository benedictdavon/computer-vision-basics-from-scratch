from __future__ import annotations

import numpy as np


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten images to vectors.

    Args:
        images: (B, H, W, C)

    Returns:
        flat: (B, H*W*C)
    """
    B = images.shape[0]
    return images.reshape(B, -1)


def reshape_to_tokens(flat: np.ndarray, num_tokens: int) -> np.ndarray:
    """Reshape flattened vectors into toy token sequences.

    Args:
        flat: (B, D_total)
        num_tokens: number of tokens N

    Returns:
        tokens: (B, N, D)
    """
    B, D_total = flat.shape
    if D_total % num_tokens != 0:
        raise ValueError("D_total must be divisible by num_tokens")

    D = D_total // num_tokens
    return flat.reshape(B, num_tokens, D)
