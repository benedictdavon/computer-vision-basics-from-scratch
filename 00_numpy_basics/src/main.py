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

from .ops.dtype_and_norm import per_channel_normalize, to_float01
from .ops.tensor_basics import flatten_images, reshape_to_tokens
from .ops.vectorization import mean_per_image_loop, mean_per_image_vectorized


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 42

    # Data / demo settings (stage-specific)
    batch_size: int = 8
    image_hw: tuple[int, int] = (32, 32)
    channels: int = 3

    # Token reshape demo
    num_tokens: int = 64

    # Visualization
    show_plots: bool = True

    # Debug verbosity
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
# Data creation (Stage 00: synthetic starter)
# -----------------------------
def make_synthetic_images_uint8(cfg: Config) -> np.ndarray:
    """Create a synthetic batch of uint8 images.

    Returns:
        images_uint8: (B, H, W, C) uint8
    """
    H, W = cfg.image_hw
    images_uint8 = np.random.randint(
        0, 256, size=(cfg.batch_size, H, W, cfg.channels), dtype=np.uint8
    )
    return images_uint8


# -----------------------------
# Stage pipeline (orchestration)
# -----------------------------
def run(cfg: Config) -> dict[str, Any]:
    """Run the stage demo.

    Returns:
        A dict of artifacts useful for debugging or future extensions.
    """
    set_seed(cfg.seed)

    # 1) Create / load data
    images_uint8 = make_synthetic_images_uint8(cfg)
    trace_tensor("images_uint8", images_uint8)

    # 2) Dtype conversion + normalization (ops)
    images_01 = to_float01(images_uint8)
    trace_tensor("images_01", images_01)

    images_norm, mean_c, std_c = per_channel_normalize(images_01, eps=1e-6)
    trace_tensor("mean_c", mean_c)
    trace_tensor("std_c", std_c)
    trace_tensor("images_norm", images_norm)

    # 3) Reshape demos (ops)
    flat = flatten_images(images_norm)
    trace_tensor("flat", flat)

    tokens = reshape_to_tokens(flat, num_tokens=cfg.num_tokens)
    trace_tensor("tokens", tokens)

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
