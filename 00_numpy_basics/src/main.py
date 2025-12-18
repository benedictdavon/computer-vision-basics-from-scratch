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
from .ops.layouts import (
    channel_mean,
    ensure_contiguous,
    nchw_to_nhwc,
    nhwc_to_nchw,
)
from .ops.linalg_basics import (
    center_rows,
    cosine_similarity,
    gram_matrix,
    pca_svd,
    project_onto_components,
)
from .ops.masking import (
    apply_token_mask,
    gather_visible_tokens,
    make_attention_mask,
    make_boolean_mask,
    random_mask_indices,
    restore_tokens,
)
from .ops.numerical_stability import (
    contains_nan_or_inf,
    safe_l2_normalize,
    softmax_sums_to_one,
    stable_softmax,
)
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


def run(cfg: Config) -> dict[str, Any]:
    set_seed(cfg.seed)

    # -------------------------
    # 1) Dataset statistics
    # -------------------------
    batch_iter = iter_cifar10_batches(cfg.data_dir, cfg.split)
    mean_c, std_c = compute_dataset_channel_stats(batch_iter)
    trace_tensor("dataset_mean_c", mean_c)
    trace_tensor("dataset_std_c", std_c)

    # -------------------------
    # 2) Load exploration batch
    # -------------------------
    images_uint8 = load_cifar10_images(
        cfg.data_dir, cfg.split, max_images=cfg.batch_size
    )
    trace_tensor("images_uint8", images_uint8)

    # -------------------------
    # 3) Dtype conversion
    # -------------------------
    images_01 = to_float01(images_uint8)
    trace_tensor("images_01", images_01)

    # -------------------------
    # 4) Apply dataset normalization
    # -------------------------
    images_norm = normalize_with_stats(images_01, mean_c, std_c)
    trace_tensor("images_norm", images_norm)

    # quick safety check
    if contains_nan_or_inf(images_norm):
        raise ValueError(
            "images_norm contains NaN/Inf — normalization or stats may be wrong."
        )

    # -------------------------
    # 5) Diagnostic batch stats
    # -------------------------
    batch_mean, batch_std = compute_batch_channel_stats(images_01)
    trace_tensor("batch_mean_c", batch_mean)
    trace_tensor("batch_std_c", batch_std)

    # -------------------------
    # 6) Layout demos (NHWC <-> NCHW + contiguity)
    # -------------------------
    images_nchw = nhwc_to_nchw(images_norm)
    trace_tensor("images_nchw", images_nchw)

    images_back = nchw_to_nhwc(images_nchw)
    trace_tensor("images_back", images_back)
    assert_close("nhwc->nchw->nhwc roundtrip", images_norm, images_back, atol=1e-6)

    images_nchw = ensure_contiguous(images_nchw)
    # channel stats under different layouts (should match)
    mean_nhwc = channel_mean(images_norm, layout="NHWC", keepdims=False)
    mean_nchw = channel_mean(images_nchw, layout="NCHW", keepdims=False)
    assert_close("channel_mean nhwc vs nchw", mean_nhwc, mean_nchw, atol=1e-6)

    # -------------------------
    # 7) Reshape demos (pixels -> tokens)
    # -------------------------
    flat_pixels = flatten_images(images_norm)  # (B, H*W*C)
    trace_tensor("flat_pixels", flat_pixels)

    tokens = reshape_to_tokens(flat_pixels, cfg.num_tokens)  # (B, N, D)
    trace_tensor("tokens", tokens)

    # -------------------------
    # 8) Vectorization sanity checks
    # -------------------------
    mean_loop = mean_per_image_loop(images_norm)
    mean_vec = mean_per_image_vectorized(images_norm)
    assert_close("loop vs vectorized mean", mean_loop, mean_vec, atol=1e-5)

    # -------------------------
    # 9) Numerical stability demos (softmax + L2 normalize)
    # -------------------------
    # make fake logits (B, N)
    logits = tokens.mean(axis=-1)  # (B, N)
    trace_tensor("logits", logits)

    probs = stable_softmax(logits, axis=-1)  # (B, N)
    trace_tensor("softmax_probs", probs)

    if not softmax_sums_to_one(probs, axis=-1, atol=1e-5):
        raise AssertionError(
            "softmax probabilities do not sum to 1 (check stable_softmax)."
        )

    # L2 normalize a batch of vectors (use tokens pooled to (B, D))
    pooled = tokens.mean(axis=1)  # (B, D)
    pooled_unit = safe_l2_normalize(pooled, axis=-1)
    trace_tensor("pooled_unit", pooled_unit)

    # -------------------------
    # 10) Linalg demos (cos sim, gram, PCA)
    # -------------------------
    # take one sample's tokens: (N, D)
    x = tokens[0]
    trace_tensor("x_tokens", x)

    g = gram_matrix(x)  # (N, N)
    trace_tensor("gram_NN", g)

    cos_nn = cosine_similarity(x)  # (N, N)
    trace_tensor("cosine_NN", cos_nn)

    # PCA demo on pooled embeddings (B, D)
    x_centered, mu = center_rows(pooled)
    comps, var = pca_svd(x_centered, k=2)
    proj = project_onto_components(x_centered, comps)  # (B, 2)
    trace_tensor("pca_proj", proj)
    trace_tensor("pca_var", var)

    # -------------------------
    # 11) Masking demos (MAE-style)
    # -------------------------
    n = tokens.shape[1]
    d = tokens.shape[2]
    mask_idx = random_mask_indices(n, mask_ratio=0.75, seed=cfg.seed)
    mask = make_boolean_mask(n, mask_idx)  # True=masked

    mask_token = np.zeros((d,), dtype=tokens.dtype)
    tokens_masked = apply_token_mask(tokens, mask, mask_token)
    trace_tensor("tokens_masked", tokens_masked)

    visible, vis_idx = gather_visible_tokens(tokens, mask)
    trace_tensor("visible_tokens", visible)

    restored = restore_tokens(visible, mask, mask_token)
    trace_tensor("restored_tokens", restored)

    # restored should match apply_token_mask output (same rule)
    assert_close("restore == apply_token_mask", restored, tokens_masked, atol=1e-6)

    # attention mask/bias (valid=True means can attend)
    valid = np.broadcast_to((~mask)[None, :], (tokens.shape[0], n))  # (B, N)
    attn_bias = make_attention_mask(valid)  # (B, 1, 1, N)
    trace_tensor("attn_bias", attn_bias)

    # -------------------------
    # 12) Visualization hook
    # -------------------------
    if cfg.show_plots:
        from pathlib import Path

        save_dir = Path("00_numpy_basics/outputs")

        from .viz.plots import show_heatmap, show_histograms, show_image_grid

        show_image_grid(
            images_uint8,
            title="Original uint8 images (B,H,W,C)",
            save_path=save_dir / "images_uint8.png",
        )

        show_image_grid(
            (images_01 * 255.0).astype(np.uint8),
            title="Normalized images (visualized)",
            save_path=save_dir / "images_normalized.png",
        )

        show_histograms(
            images_uint8,
            images_01,
            title="Pixel distributions",
            save_path=save_dir / "pixel_histogram.png",
        )

        show_heatmap(
            cos_nn,
            title="Token Cosine Similarity (NxN)",
            cmap="coolwarm",
            save_path=save_dir / "token_cosine_similarity.png",
        )

        show_heatmap(
            g,
            title="Token Gram Matrix",
            cmap="magma",
            save_path=save_dir / "token_gram_matrix.png",
        )

    return {
        "images_uint8": images_uint8,
        "images_01": images_01,
        "images_norm": images_norm,
        "tokens": tokens,
        "tokens_masked": tokens_masked,
        "attn_bias": attn_bias,
        "pca_proj": proj,
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
