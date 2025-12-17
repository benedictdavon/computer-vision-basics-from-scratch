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


def var_per_image_loop(images: np.ndarray) -> np.ndarray:
    """Compute per-image variance using an explicit Python loop.

    Args:
        images: (B, H, W, C)

    Returns:
        variances: (B,)
    """
    B = images.shape[0]
    variances = np.zeros((B,), dtype=np.float32)
    for i in range(B):
        variances[i] = float(np.var(images[i]))
    return variances


def var_per_image_vectorized(images: np.ndarray) -> np.ndarray:
    """Compute per-image variance using vectorized NumPy ops.

    Args:
        images: (B, H, W, C)

    Returns:
        variances: (B,)
    """
    mean = images.mean(axis=(1, 2, 3), keepdims=True)
    var = ((images - mean) ** 2).mean(axis=(1, 2, 3))
    return var.astype(np.float32)


def gamma_correction_loop(images: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction using an explicit Python loop.

    Args:
        images: (B, H, W, C) float32 in [0, 1]
        gamma: float

    Returns:
        images: (B, H, W, C)
    """
    B, H, W, C = images.shape
    corrected = np.zeros_like(images, dtype=np.float32)
    for i in range(B):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    corrected[i, h, w, c] = images[i, h, w, c] ** gamma
    return corrected


def gamma_correction_vectorized(images: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction using vectorized NumPy ops.

    Args:
        images: (B, H, W, C) float32 in [0, 1]
        gamma: float

    Returns:
        images: (B, H, W, C)
    """
    return np.power(images, gamma).astype(np.float32)


def pairwise_sq_distance_loop(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances using an explicit Python loop.

    Args:
        X: (N, D)
        Y: (M, D)

    Returns:
        distances: (N, M)
    """
    N = x.shape[0]
    M = y.shape[0]
    distances = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            diff = x[i] - y[j]
            distances[i, j] = float(np.sum(diff**2))
    return distances


def pairwise_sq_distance_vectorized(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances using vectorized NumPy ops.

    Args:
        X: (N, D)
        Y: (M, D)

    Returns:
        distances: (N, M)
    """
    X_sq = np.sum(x**2, axis=1, keepdims=True)  # (N, 1)
    Y_sq = np.sum(y**2, axis=1)  # (M,)
    cross_term = np.dot(x, y.T)  # (N, M)
    # X - 2 * X@Y.T + Y
    distances = X_sq - 2 * cross_term + Y_sq  # Broadcasting
    return distances.astype(np.float32)


def euclidean_distance_from_sq(dist_sq: np.ndarray) -> np.ndarray:
    return np.sqrt(dist_sq)
