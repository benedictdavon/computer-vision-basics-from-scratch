# # main.py (Stage Template)
# #
# # Rules:
# # - Orchestrator only: no core math here.
# # - All reusable logic must live in ops/, data/, viz/, layers/, models/, etc.
# # - Always print shape traces and run at least one sanity check.
# # - Produce at least one visualization when relevant.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data.cifar10 import (
    compute_batch_channel_stats,
    compute_dataset_channel_stats,
    iter_cifar10_batches,
    load_cifar10_images,
    normalize_with_stats,
)
from .ops.dtype_and_norm import to_float01
from .ops.tensor_basics import flatten_images, reshape_to_tokens
from .ops.vectorization import mean_per_image_loop, mean_per_image_vectorized


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 42

    # Dataset
    data_dir: str = ".data"
    split: str = "train"
    batch_size: int = 8

    # Token reshape demo
    num_tokens: int = 64

    # Visualization
    show_plots: bool = True
    verbose: bool = True


# -----------------------------
# Logging / tracing helpers
# -----------------------------
def trace_tensor(name: str, x: np.ndarray) -> None:
    """Print a compact tensor trace: shape, dtype, min/max/mean.

    Args:
        name: Human-readable tensor name.
        x: NumPy array.

    Notes:
        Use this after every meaningful transformation in main.py.
    """
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_mean = float(np.mean(x))
    print(
        f"[trace] {name:<18} shape={x.shape!s:<16} dtype={x.dtype} "
        f"min={x_min:.4g} max={x_max:.4g} mean={x_mean:.4g}"
    )


def assert_close(name: str, a: np.ndarray, b: np.ndarray, atol: float = 1e-6) -> None:
    """Sanity check for numerical equivalence."""
    ok = np.allclose(a, b, atol=atol, rtol=0.0)
    if not ok:
        diff = float(np.max(np.abs(a - b)))
        raise AssertionError(
            f"[check] {name} failed: max|a-b|={diff:.4g} (atol={atol})"
        )
    print(f"[check] {name} passed (atol={atol})")


def set_seed(seed: int) -> None:
    np.random.seed(seed)


# -----------------------------
# Stage pipeline (orchestration)
# -----------------------------
def run(cfg: Config) -> dict[str, Any]:
    """Run the stage demo.

    Returns:
        A dict of artifacts useful for debugging or future extensions.
    """
    set_seed(cfg.seed)

    # 1) dataset statistics
    batch_iter = iter_cifar10_batches(cfg.data_dir, cfg.split)
    mean_c, std_c = compute_dataset_channel_stats(batch_iter)
    trace_tensor("dataset_mean_c", mean_c)
    trace_tensor("dataset_std_c", std_c)

    # 2) load exploration batch
    images_uint8 = load_cifar10_images(
        cfg.data_dir,
        cfg.split,
        max_images=cfg.batch_size,
    )
    trace_tensor("images_uint8", images_uint8)

    # 3) dtype conversion
    images_01 = to_float01(images_uint8)
    trace_tensor("images_01", images_01)

    # 4) apply dataset normalization
    images_norm = normalize_with_stats(images_01, mean_c, std_c)
    trace_tensor("images_norm", images_norm)

    # 5) diagnostic batch stats (optional)
    batch_mean, batch_std = compute_batch_channel_stats(images_01)
    trace_tensor("batch_mean_c", batch_mean)
    trace_tensor("batch_std_c", batch_std)

    # 6) reshaping demos
    flat_pixels = flatten_images(images_norm)
    tokens = reshape_to_tokens(flat_pixels, cfg.num_tokens)

    # 4) Vectorization vs loop sanity (ops)
    mean_loop = mean_per_image_loop(images_norm)
    mean_vec = mean_per_image_vectorized(images_norm)
    assert_close("loop vs vectorized mean", mean_loop, mean_vec, atol=1e-5)

    # 5) Visualization hook
    if cfg.show_plots:
        from .viz.plots import show_histograms, show_image_grid

        show_image_grid(images_uint8, title="Original uint8 images (B,H,W,C)")

        # For visualization, we can display images_01 scaled back to uint8.
        show_image_grid(
            (images_01 * 255.0).astype(np.uint8), title="Normalized images (visualized)"
        )

        show_histograms(
            images_uint8, images_01, title="Pixel distributions: uint8 vs normalized"
        )

    return {
        "images_uint8": images_uint8,
        "images_01": images_01,
        "images_norm": images_norm,
        "tokens": tokens,
    }


def main() -> None:
    cfg = Config()
    if cfg.verbose:
        print(f"[info] Running Stage 00 with config: {cfg}")
    run(cfg)
    if cfg.verbose:
        print("[info] Done.")


if __name__ == "__main__":
    main()
