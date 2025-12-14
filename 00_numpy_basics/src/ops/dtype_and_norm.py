from __future__ import annotations

import numpy as np


def to_float01(images_uint8: np.ndarray) -> np.ndarray:
    """Convert uint8 images to float32 in [0, 1].

    Args:
        images_uint8: (B, H, W, C) uint8

    Returns:
        images_01: (B, H, W, C) float32 in [0, 1]
    """
    if images_uint8.dtype != np.uint8:
        raise TypeError("Expected uint8 input")

    images_f32 = images_uint8.astype(np.float32)
    return images_f32 / 255.0


def per_channel_normalize(
    images: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel normalization with broadcasting.

    Args:
        images: (B, H, W, C) float array
        eps: numerical stability epsilon

    Returns:
        images_norm: (B, H, W, C)
        mean_c: (C,)
        std_c: (C,)
    """
    mean_c = images.mean(axis=(0, 1, 2))
    std_c = images.std(axis=(0, 1, 2)) + eps
    images_norm = (images - mean_c) / std_c
    return images_norm, mean_c, std_c
