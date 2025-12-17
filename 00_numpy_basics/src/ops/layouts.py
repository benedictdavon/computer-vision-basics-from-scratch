from __future__ import annotations

import numpy as np


def nhwc_to_nchw(images: np.ndarray) -> np.ndarray:
    """
    Convert images from NHWC to NCHW layout.
    Args:
        images: (N, H, W, C) NumPy array.

    Returns:
        images: (N, C, H, W) NumPy array.
    """
    if images.ndim != 4:
        raise ValueError("Expected images with shape (N, H, W, C)")

    return np.transpose(images, (0, 3, 1, 2))


def nchw_to_nhwc(images: np.ndarray) -> np.ndarray:
    """
    Convert images from NCHW to NHWC layout.
    Args:
        images: (N, C, H, W) NumPy array.

    Returns:
        images: (N, H, W, C) NumPy array.
    """
    if images.ndim != 4:
        raise ValueError("Expected images with shape (N, C, H, W)")

    return np.transpose(images, (0, 2, 3, 1))


def channel_mean(
    images: np.ndarray,
    layout: str = "NHWC",
    keepdims: bool = False,
) -> np.ndarray:
    """
    Compute per-channel mean over batch and spatial dimensions.
    Args:
        images: (B, H, W, C) or (B, C, H, W) NumPy array.
        layout: "NHWC" or "NCHW".
        keepdims: If True, keep reduced dimensions.

    Returns:
        mean_c: (C,) or broadcast shape depending on keepdims.
    """
    if layout not in ("NHWC", "NCHW"):
        raise ValueError("layout must be either 'NHWC' or 'NCHW'")
    if images.ndim != 4:
        if layout == "NHWC":
            raise ValueError("Expected images with shape (B, H, W, C)")
        raise ValueError("Expected images with shape (B, C, H, W)")

    if layout == "NHWC":
        return images.mean(axis=(0, 1, 2), keepdims=keepdims)

    return images.mean(axis=(0, 2, 3), keepdims=keepdims)


def channel_std(
    images: np.ndarray,
    layout: str = "NHWC",
    keepdims: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute per-channel std over batch and spatial dimensions.
    Args:
        images: (B, H, W, C) or (B, C, H, W) NumPy array.
        layout: "NHWC" or "NCHW".
        keepdims: If True, keep reduced dimensions.

    Returns:
        std_c: (C,) or broadcast shape depending on keepdims.
    """
    if layout not in ("NHWC", "NCHW"):
        raise ValueError("layout must be either 'NHWC' or 'NCHW'")
    if images.ndim != 4:
        if layout == "NHWC":
            raise ValueError("Expected images with shape (B, H, W, C)")
        raise ValueError("Expected images with shape (B, C, H, W)")

    if layout == "NHWC":
        return images.std(axis=(0, 1, 2), keepdims=keepdims) + eps

    return images.std(axis=(0, 2, 3), keepdims=keepdims) + eps


def center_by_channel_mean(
    images: np.ndarray,
    mean_c: np.ndarray,
    layout: str = "NHWC",
) -> np.ndarray:
    """
    Center images by subtracting per-channel mean.
    Args:
        images: (B, H, W, C) or (B, C, H, W) NumPy array.
        mean_c: (C,) or broadcast shape NumPy array.
        layout: "NHWC" or "NCHW".

    Returns:
        centered_images: same shape as images.
    """
    if layout not in ("NHWC", "NCHW"):
        raise ValueError("layout must be either 'NHWC' or 'NCHW'")
    if images.ndim != 4:
        if layout == "NHWC":
            raise ValueError("Expected images with shape (B, H, W, C)")
        raise ValueError("Expected images with shape (B, C, H, W)")

    c = images.shape[3] if layout == "NHWC" else images.shape[1]

    if mean_c.ndim == 1:
        if mean_c.shape != (c,):
            raise ValueError(f"Expected mean_c with shape ({c},)")
        return images - mean_c

    if mean_c.ndim == 4:
        if layout == "NHWC":
            if mean_c.shape != (1, 1, 1, c):
                raise ValueError(f"Expected mean_c with shape (1, 1, 1, {c})")
        else:
            if mean_c.shape != (1, c, 1, 1):
                raise ValueError(f"Expected mean_c with shape (1, {c}, 1, 1)")
        return images - mean_c

    raise ValueError(
        "Expected mean_c with shape (C,) or broadcast shape for the layout"
    )


def is_contiguous(x: np.ndarray) -> bool:
    """
    Check if a NumPy array is stored in contiguous memory.
    Args:
        x: NumPy array.
    Returns:
        True if x is contiguous, False otherwise.
    """
    return x.flags["C_CONTIGUOUS"]


def ensure_contiguous(x: np.ndarray) -> np.ndarray:
    """
    Ensure a NumPy array is stored in contiguous memory.
    Args:
        x: NumPy array.
    Returns:
        Contiguous NumPy array.
    """
    if not is_contiguous(x):
        return np.ascontiguousarray(x)
    return x
