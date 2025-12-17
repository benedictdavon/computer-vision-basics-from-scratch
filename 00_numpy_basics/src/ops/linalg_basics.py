from __future__ import annotations

import numpy as np

"""
Linear Algebra Primitives for Computer Vision

This module implements NumPy-only building blocks that appear throughout
modern CV pipelines:

- L2 normalization → cosine similarity, attention
- Gram matrices → token similarity, style loss
- PCA via SVD → dimensionality reduction, whitening

All implementations are explicit and shape-aware for learning purposes.
"""


def l2_norm(
    x: np.ndarray,
    axis: int = -1,
    keepdims: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute L2 norm along an axis with numerical stability.

    Args:
        x: Input array.
        axis: Axis to reduce over.
        keepdims: Keep reduced dimensions for broadcasting.
        eps: Small constant added inside sqrt to avoid zero division downstream.

    Returns:
        L2 norm of x.
    """
    sum_sq = np.sum(x * x, axis=axis, keepdims=keepdims)
    return np.sqrt(sum_sq + eps)


def gram_matrix(x: np.ndarray) -> np.ndarray:
    """Compute the row-wise Gram matrix (similarity) for x.

    For x shaped (N, D), returns (N, N) via x @ x.T.
    This matches attention-style score computation (Q @ K.T) without softmax.

    Args:
        x: (N, D) array.

    Returns:
        (N, N) Gram matrix.
    """
    if x.ndim != 2:
        raise ValueError("Expected x with shape (N, D)")
    return x @ x.T


def cosine_similarity(
    x: np.ndarray,
    y: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Cosine similarity supporting vector and matrix inputs.

    Cases:
    - Vector-vector:
        x: (D,), y: (D,) -> scalar ()
    - Matrix-matrix (cross-similarity):
        x: (N, D), y: (M, D) -> (N, M)
    - Matrix only (self-similarity):
        x: (N, D), y=None -> (N, N)

    Args:
        x: (D,) or (N, D)
        y: (D,) or (M, D) or None
        eps: Small constant for stability.

    Returns:
        Cosine similarity array (shape depends on input case).
    """
    if y is None:
        y = x

    if x.ndim == 1 and y.ndim == 1:
        if x.shape != y.shape:
            raise ValueError("For vector inputs, x and y must have the same shape (D,)")
        dot = float(np.dot(x, y))
        nx = float(l2_norm(x, axis=0, keepdims=False, eps=eps))
        ny = float(l2_norm(y, axis=0, keepdims=False, eps=eps))
        return np.array(dot / (nx * ny), dtype=np.float64)

    if x.ndim == 2 and y.ndim == 2:
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                "For matrix inputs, x and y must share the same feature dim D"
            )

        x_norm = l2_norm(x, axis=1, keepdims=True, eps=eps)  # (N, 1)
        y_norm = l2_norm(y, axis=1, keepdims=True, eps=eps)  # (M, 1)

        x_unit = x / x_norm  # (N, D)
        y_unit = y / y_norm  # (M, D)

        return x_unit @ y_unit.T  # (N, M)

    raise ValueError("x and y must both be 1D or both be 2D")


def outer_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the outer product of two vectors.
    Outer product:
        O = u * v^T

    Args:
        u: Input vector of shape (M,).
        v: Input vector of shape (N,).

    Returns:
        Outer product matrix of shape (M, N).
    """
    if u.ndim != 1 or v.ndim != 1:
        raise ValueError("Both u and v must be 1-dimensional arrays.")

    return np.outer(u, v)


# PCA & SVD helpers
def center_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Center the rows of the input array by subtracting the mean of each column.
    Args:
        x: Input NumPy array of shape (N, D), where N is the number of samples and D is the feature dimension.

    Returns:
        x_centered: Centered array of shape (N, D).
        mean: Mean of each column of shape (D,).
    """
    mean = np.mean(x, axis=0)
    x_centered = x - mean
    return x_centered, mean


def pca_svd(x_centered: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform PCA using SVD on the centered input array.
    PCA:
        Find the top-k principal components of the data.
    SVD:
        X = U * S * V^T

    Args:
        x_centered: Centered input NumPy array of shape (N, D).
        k: Number of principal components to return.

    Returns:
        components: Principal components of shape (k, D).
        explained_variance: Explained variance for each principal component of shape (k,).
    """
    # SVD: X = U S V^T
    # Principal components are rows of V^T
    # Variance explained by each component ∝ S^2

    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:k]
    explained_variance = (s[:k] ** 2) / (x_centered.shape[0] - 1)
    return components, explained_variance


def project_onto_components(
    x_centered: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """
    Project the centered input array onto the principal components.
    Args:
        x_centered: Centered input NumPy array of shape (N, D).
        components: Principal components of shape (k, D).

    Returns:
        projected_data: Projected data of shape (N, k).
    """
    return np.dot(x_centered, components.T)
