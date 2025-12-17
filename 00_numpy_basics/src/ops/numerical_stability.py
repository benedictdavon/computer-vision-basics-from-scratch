from __future__ import annotations

import numpy as np


def contains_nan_or_inf(x: np.ndarray) -> bool:
    """Check if the array contains any NaN or Inf values.

    Args:
        x: Input NumPy array.
    Returns:
        True if NaN or Inf values are present, False otherwise.
    """
    return np.isnan(x).any() or np.isinf(x).any()


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the softmax of the input array in a numerically stable way.
    Softmax:
        softmax(x_i) = exp(x_i) / sum_j exp(x_j)

    Args:
        x: Input NumPy array.
        axis: Axis along which to compute the softmax.
    Returns:
        Softmax of the input array.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    sum_e_x = np.sum(e_x, axis=axis, keepdims=True)
    # Avoid division by zero
    sum_e_x = np.maximum(sum_e_x, 1e-12)
    return e_x / sum_e_x


def softmax_sums_to_one(p: np.ndarray, axis: int = -1, atol: float = 1e-6) -> bool:
    """
    Check if the softmax probabilities sum to one along the specified axis.
    Args:
        p: Softmax probabilities.
        axis: Axis along which to check the sum.
        atol: Absolute tolerance for the sum check.
    Returns:
        True if sums are close to one, False otherwise.
    """
    sums = np.sum(p, axis=axis, keepdims=True)
    ones = np.ones_like(sums)
    return np.allclose(sums, ones, atol=atol)


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Safely divide two arrays, avoiding division by zero.

    Args:
        a: Numerator array.
        b: Denominator array.
        eps: Small constant to add to denominator for stability.
    Returns:
        Result of the division.
    """
    return a / (b + eps)


def safe_l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    """
    L2 normalize the input array along the specified axis in a numerically stable way.

    Args:
        x: Input NumPy array.
        axis: Axis along which to normalize.
        eps: Small constant to avoid division by zero.
    Returns:
        L2 normalized array.
    """
    norm = np.sqrt(np.sum(x**2, axis=axis, keepdims=True))
    return x / (norm + eps)
