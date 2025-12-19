from __future__ import annotations

import numpy as np


def to_float01(x: np.ndarray) -> np.ndarray:
    """
    Strict conversion to float32 in [0,1].

    Accepts ONLY:
      - uint8 images in [0,255]
      - float images already in [0,1]

    Rejects:
      - float images in [0,255]
      - negative values
      - NaN / Inf
    """
    assert_image_ndim(x)

    # uint8 input
    if x.dtype == np.uint8:
        x_f = x.astype(np.float32) / 255.0
        # This should always pass, but keeps the contract explicit
        assert_range_float01(x_f)
        return x_f

    # float input
    if x.dtype in (np.float32, np.float64):
        x_f = x.astype(np.float32, copy=False)

        # Strict: must already be [0,1]
        assert_range_float01(x_f)
        return x_f

    raise TypeError(f"to_float01 expects uint8 or float array, got {x.dtype}")


def to_uint8(x: np.ndarray, *, assume_float01: bool = True) -> np.ndarray:
    """
    Strict conversion from float to uint8.

    If assume_float01=True:
      - x must be float in [0,1]

    If assume_float01=False:
      - x must be float in [0,255]
    """
    assert_dtype(x, kind="float")

    x_f = x.astype(np.float32, copy=False)

    if not np.isfinite(x_f).all():
        raise ValueError("Float image contains NaN or Inf")

    if assume_float01:
        # STRICT: must already be [0,1]
        assert_range_float01(x_f)
        x_f = x_f * 255.0
    else:
        min_val = x_f.min()
        max_val = x_f.max()
        if min_val < 0.0 or max_val > 255.0:
            raise ValueError(
                f"Float image expected in [0,255], "
                f"but got range [{min_val:.3f}, {max_val:.3f}]"
            )

    return np.round(x_f).astype(np.uint8)


def ensure_channel_last(x: np.ndarray) -> np.ndarray:
    """
    Ensure image uses channel-last layout.

    Rules:
      - (H, W)       -> (H, W, 1)
      - (H, W, C)    -> unchanged (C must be 1,3,4)
      - (B, H, W, C) -> unchanged (C must be 1,3,4)
    """
    assert_image_ndim(x)

    if x.ndim == 2:
        return x[..., None]

    # ndim is 3 or 4 here
    assert_channel_dim(x)
    return x


def squeeze_gray_channel(x: np.ndarray) -> np.ndarray:
    """
    Remove trailing singleton channel for grayscale images.

    Example:
      (H, W, 1) -> (H, W)

    If no singleton channel exists, return input unchanged.
    """
    assert_image_ndim(x)

    if x.ndim == 3 and x.shape[-1] == 1:
        return x[..., 0]
    return x


def assert_image_ndim(
    x: np.ndarray,
    *,
    allowed: tuple[int, ...] = (2, 3, 4),
) -> None:
    """
    Assert that image has an allowed number of dimensions.
    """
    if x.ndim not in allowed:
        raise ValueError(f"Invalid image ndim: got {x.ndim}, expected one of {allowed}")


def assert_channel_dim(
    x: np.ndarray,
    *,
    allowed_c: tuple[int, ...] = (1, 3, 4),
) -> None:
    """
    Assert that channel dimension (if present) is valid.
    """
    if x.ndim >= 3:
        c = x.shape[-1]
        if c not in allowed_c:
            raise ValueError(
                f"Invalid channel dimension: C={c}, expected one of {allowed_c}"
            )


def assert_dtype(
    x: np.ndarray,
    *,
    kind: str,
) -> None:
    """
    Assert dtype category.

    kind:
      - "uint8"
      - "float"
    """
    if kind == "uint8":
        if x.dtype != np.uint8:
            raise TypeError(f"Expected uint8 array, got {x.dtype}")
        return

    if kind == "float":
        if x.dtype not in (np.float32, np.float64):
            raise TypeError(f"Expected float array, got {x.dtype}")
        return

    raise ValueError(f"Unknown dtype kind: {kind}")


def assert_range_uint8(x: np.ndarray) -> None:
    """
    Assert uint8 image range contract.
    """
    if x.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {x.dtype}")

    # Redundant for uint8, but useful if something was cast incorrectly
    if x.min() < 0 or x.max() > 255:
        raise ValueError(f"uint8 image out of range: [{x.min()}, {x.max()}]")


def assert_range_float01(
    x: np.ndarray,
    *,
    eps: float = 1e-6,
) -> None:
    """
    Assert float image range contract [0,1].
    """
    if x.dtype not in (np.float32, np.float64):
        raise TypeError(f"Expected float image, got {x.dtype}")

    if not np.isfinite(x).all():
        raise ValueError("Float image contains NaN or Inf")

    min_val = x.min()
    max_val = x.max()

    if min_val < -eps or max_val > 1.0 + eps:
        raise ValueError(
            f"Float image expected in [0,1], "
            f"but got range [{min_val:.3f}, {max_val:.3f}]"
        )


def describe(x: np.ndarray, *, name: str = "x") -> str:
    """
    Return a formatted string describing the signal.

    Includes:
      - shape
      - dtype
      - min / max / mean / std
      - per-channel stats if applicable
    """
    lines = []
    lines.append(f"{name}:")
    lines.append(f"  shape: {x.shape}")
    lines.append(f"  dtype: {x.dtype}")
    lines.append(f"  min/max: {x.min():.4f} / {x.max():.4f}")
    lines.append(f"  mean/std: {x.mean():.4f} / {x.std():.4f}")

    if x.ndim >= 3:
        c = x.shape[-1]
        if c in (1, 3, 4):
            mean_c = x.mean(axis=tuple(range(x.ndim - 1)))
            std_c = x.std(axis=tuple(range(x.ndim - 1)))
            lines.append(f"  per-channel mean: {mean_c}")
            lines.append(f"  per-channel std:  {std_c}")

    return "\n".join(lines)
