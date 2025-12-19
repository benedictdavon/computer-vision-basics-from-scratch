from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SampleImages:
    """
    Container for small image batches used in demos.

    images_uint8:
      - shape: (B, H, W, C)
      - dtype: uint8
      - range: [0,255]
    """

    images_uint8: np.ndarray  # (B, H, W, C), uint8 in [0,255]
    labels: np.ndarray | None = None  # CIFAR loader currently has no labels API


def _load_stage00_cifar10_module(repo_root: Path):
    """
    Load Stage 00 CIFAR-10 loader via file path because folder name
    starts with digits (00_numpy_basics), which is not importable
    as a normal Python package.
    """
    cifar_path = repo_root / "00_numpy_basics" / "src" / "data" / "cifar10.py"
    if not cifar_path.exists():
        raise FileNotFoundError(f"Stage 00 CIFAR loader not found at: {cifar_path}")

    spec = importlib.util.spec_from_file_location("stage00_cifar10", str(cifar_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for: {cifar_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def load_sample_images_cifar10(
    batch_size: int = 16,
    *,
    data_dir: str | Path = ".data",
    split: str = "train",
    seed: int = 0,
    shuffle: bool = True,
) -> SampleImages:
    """
    Load a small batch of CIFAR-10 images using Stage 00 loader.

    This function intentionally returns ONLY raw uint8 images.
    Representation conversion (float, normalization, etc.)
    must be done explicitly in demo / main code using ops helpers.

    Args:
        batch_size: number of images to return
        data_dir: dataset cache directory (same as Stage 00)
        split: "train" or "test"
        seed: RNG seed for sampling
        shuffle: whether to randomly sample

    Returns:
        SampleImages with:
          - images_uint8: (B, 32, 32, 3) uint8 in [0,255]
    """
    # Repo root = .../computer-vision-basics-from-scratch
    repo_root = Path(__file__).resolve().parents[3]
    cifar = _load_stage00_cifar10_module(repo_root)

    images_uint8_all = cifar.load_cifar10_images(
        data_dir=data_dir,
        split=split,
        max_images=None,
    )  # (N, 32, 32, 3) uint8

    # Strict contract checks
    if images_uint8_all.ndim != 4:
        raise ValueError(
            f"Expected CIFAR images with ndim=4, got {images_uint8_all.ndim}"
        )
    if images_uint8_all.shape[-1] != 3:
        raise ValueError(
            f"Expected CIFAR images with 3 channels, got shape {images_uint8_all.shape}"
        )
    if images_uint8_all.dtype != np.uint8:
        raise TypeError(f"Expected uint8 CIFAR images, got {images_uint8_all.dtype}")

    n_total = images_uint8_all.shape[0]
    if batch_size <= 0 or batch_size > n_total:
        raise ValueError(f"batch_size must be in [1, {n_total}], got {batch_size}")

    # sampling
    indices = np.arange(n_total)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    indices = indices[:batch_size]
    images_uint8 = images_uint8_all[indices]

    return SampleImages(images_uint8=images_uint8)
