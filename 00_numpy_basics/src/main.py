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


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 42

    # Data / demo settings (stage-specific)
    batch_size: int = 8

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
    # Use float conversion carefully for readable stats
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

    # -------------------------
    # 1) Create / load data (Stage 00: start with synthetic first)
    #    Later you can swap this with data.cifar10 / data.mnist.
    # -------------------------
    # Example: fake "image batch" in uint8 like real images
    # images_uint8: (B, H, W, C)
    B, H, W, C = cfg.batch_size, 32, 32, 3
    images_uint8 = np.random.randint(0, 256, size=(B, H, W, C), dtype=np.uint8)
    trace_tensor("images_uint8", images_uint8)

    # -------------------------
    # 2) Dtype conversion
    # -------------------------
    images_f32 = images_uint8.astype(np.float32)
    trace_tensor("images_f32", images_f32)

    # -------------------------
    # 3) Normalization demos (keep logic minimal here; move to ops later)
    #    In Stage 00, you may temporarily keep a tiny normalization snippet.
    #    But once ops/dtype_and_norm.py exists, call into it.
    # -------------------------
    images_01 = images_f32 / 255.0
    trace_tensor("images_01", images_01)

    # Per-channel mean (broadcast over H,W)
    # mean_c: (C,)
    mean_c = images_01.mean(axis=(0, 1, 2))
    # std_c: (C,)
    std_c = images_01.std(axis=(0, 1, 2)) + 1e-6
    trace_tensor("mean_c", mean_c)
    trace_tensor("std_c", std_c)

    # Broadcast: (B,H,W,C) - (C,) -> (B,H,W,C)
    images_norm = (images_01 - mean_c) / std_c
    trace_tensor("images_norm", images_norm)

    # -------------------------
    # 4) Reshape demos (bridge to later tokens)
    # -------------------------
    flat = images_norm.reshape(B, H * W * C)  # (B, H*W*C)
    trace_tensor("flat", flat)

    # Toy "tokens": split flattened into N tokens of dimension D
    # Here we just reshape for intuition.
    N = 64
    D = (H * W * C) // N
    tokens = flat[:, : N * D].reshape(B, N, D)  # (B, N, D)
    trace_tensor("tokens", tokens)

    # -------------------------
    # 5) Vectorization vs loop sanity (toy example)
    # -------------------------
    # Example: per-image mean using loop vs vectorized
    mean_loop = np.zeros((B,), dtype=np.float32)
    for i in range(B):
        mean_loop[i] = float(np.mean(images_norm[i]))
    mean_vec = images_norm.mean(axis=(1, 2, 3)).astype(np.float32)

    assert_close("loop vs vectorized mean", mean_loop, mean_vec, atol=1e-5)

    # -------------------------
    # 6) Visualization hook
    # -------------------------
    if cfg.show_plots:
        # Keep the import local so matplotlib is optional for non-plot runs.
        from .viz.plots import show_histograms, show_image_grid  # type: ignore

        # show original and normalized images
        show_image_grid(images_uint8, title="Original uint8 images (B,H,W,C)")
        # For visualization, bring to [0,1] range:
        vis_norm = images_01  # (B,H,W,C) in [0,1]
        show_image_grid(
            (vis_norm * 255.0).astype(np.uint8), title="Normalized images (visualized)"
        )

        show_histograms(
            images_uint8, images_01, title="Pixel distributions: uint8 vs normalized"
        )

    return {
        "images_uint8": images_uint8,
        "images_f32": images_f32,
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
