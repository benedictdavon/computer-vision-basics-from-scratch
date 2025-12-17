from __future__ import annotations

import pickle
import tarfile
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DIRNAME = "cifar-10-batches-py"


# -----------------------------
# Download / extraction
# -----------------------------
def ensure_cifar10_extracted(data_dir: str | Path = ".data") -> Path:
    """Ensure CIFAR-10 (python version) is downloaded and extracted.

    Args:
        data_dir: Directory used to store the downloaded tar.gz and extracted files.

    Returns:
        Path to extracted CIFAR-10 directory (cifar-10-batches-py).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_dir / "cifar-10-python.tar.gz"
    extract_dir = data_dir / CIFAR10_DIRNAME

    if extract_dir.exists():
        return extract_dir

    if not tar_path.exists():
        print("[data] Downloading CIFAR-10...")
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)

    print("[data] Extracting CIFAR-10...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    if not extract_dir.exists():
        raise FileNotFoundError(
            f"Expected extracted directory not found: {extract_dir}"
        )

    return extract_dir


def _batch_paths(cifar_dir: Path, split: str) -> list[Path]:
    if split == "train":
        return [cifar_dir / f"data_batch_{i}" for i in range(1, 6)]
    if split == "test":
        return [cifar_dir / "test_batch"]
    raise ValueError("split must be 'train' or 'test'")


def _load_batch_images_uint8(batch_path: Path) -> np.ndarray:
    """Load a single CIFAR-10 batch file to images.

    Returns:
        images_uint8: (10000, 32, 32, 3) uint8
    """
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    data = batch[b"data"]  # (10000, 3072) uint8
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
    return images.astype(np.uint8)


# -----------------------------
# Public data access APIs
# -----------------------------
def iter_cifar10_batches(
    data_dir: str | Path = ".data",
    split: str = "train",
) -> Iterator[np.ndarray]:
    """Yield CIFAR-10 batches as uint8 NHWC arrays.

    Args:
        data_dir: Dataset cache directory.
        split: "train" or "test".

    Yields:
        images_uint8: (B, 32, 32, 3) uint8 (B is typically 10000).
    """
    cifar_dir = ensure_cifar10_extracted(data_dir)
    for path in _batch_paths(cifar_dir, split):
        yield _load_batch_images_uint8(path)


def load_cifar10_images(
    data_dir: str | Path = ".data",
    split: str = "train",
    max_images: int | None = None,
) -> np.ndarray:
    """Load CIFAR-10 images into a single array (may use more memory).

    Args:
        data_dir: Dataset cache directory.
        split: "train" or "test".
        max_images: Optional cap for quick experiments.

    Returns:
        images_uint8: (N, 32, 32, 3) uint8
    """
    batches = list(iter_cifar10_batches(data_dir=data_dir, split=split))
    images = np.concatenate(batches, axis=0)
    if max_images is not None:
        images = images[:max_images]
    return images


# -----------------------------
# Statistics (estimation)
# -----------------------------
def compute_dataset_channel_stats(
    batch_iter: Iterable[np.ndarray],
    *,
    scale_to_unit: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dataset-level per-channel mean/std via streaming accumulation.

    This is the correct method for full-dataset statistics:
    - batch-size independent
    - memory safe
    - deterministic

    Args:
        batch_iter: Iterable yielding batches of images as uint8 NHWC arrays.
        scale_to_unit: If True, compute stats on float images in [0,1] (divide by 255).

    Returns:
        mean_c: (3,) float64
        std_c: (3,) float64
    """
    sum_c = np.zeros((3,), dtype=np.float64)
    sum_sq_c = np.zeros((3,), dtype=np.float64)
    count = 0  # number of pixels per channel

    for images_uint8 in batch_iter:
        if images_uint8.dtype != np.uint8:
            raise TypeError("Expected uint8 images from batch iterator")

        x = images_uint8.astype(np.float32)
        if scale_to_unit:
            x = x / 255.0

        # x: (B, H, W, C)
        sum_c += x.sum(axis=(0, 1, 2))
        sum_sq_c += (x * x).sum(axis=(0, 1, 2))

        B, H, W, C = x.shape
        if C != 3:
            raise ValueError("Expected 3 channels for CIFAR-10")
        count += B * H * W

    if count == 0:
        raise ValueError("Empty dataset iterator: no images seen")

    mean_c = sum_c / count
    var_c = (sum_sq_c / count) - (mean_c * mean_c)
    std_c = np.sqrt(np.maximum(var_c, 0.0))
    return mean_c, std_c


def compute_batch_channel_stats(
    images: np.ndarray,
    *,
    scale_to_unit: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std within a single batch (diagnostic only).

    Args:
        images: (B, H, W, C) uint8 or float.
        scale_to_unit: If True and images are uint8, convert to float in [0,1].

    Returns:
        mean_c: (C,)
        std_c: (C,)
    """
    if images.ndim != 4:
        raise ValueError("Expected images with shape (B, H, W, C)")

    x = images
    if x.dtype == np.uint8 and scale_to_unit:
        x = x.astype(np.float32) / 255.0

    mean_c = x.mean(axis=(0, 1, 2))
    std_c = x.std(axis=(0, 1, 2))
    return mean_c, std_c


# -----------------------------
# Normalization (application)
# -----------------------------
def normalize_with_stats(
    images: np.ndarray,
    mean_c: np.ndarray,
    std_c: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize images using provided per-channel statistics.

    Args:
        images: (..., C) float array.
        mean_c: (C,)
        std_c: (C,)
        eps: Stability term to avoid division by zero.

    Returns:
        normalized images with same shape as input.
    """
    if images.ndim < 1:
        raise ValueError("images must be an array")
    if mean_c.shape != std_c.shape:
        raise ValueError("mean_c and std_c must have the same shape")

    return (images - mean_c) / (std_c + eps)
