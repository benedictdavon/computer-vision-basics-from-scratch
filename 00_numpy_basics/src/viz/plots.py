from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def show_image_grid(
    images_uint8: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    """Show (and optionally save) a grid of images.

    Args:
        images_uint8: (B, H, W, C) uint8 images.
        title: Plot title.
        save_path: Optional path to save the figure.
    """
    if images_uint8.ndim != 4:
        raise ValueError("Expected images_uint8 with shape (B, H, W, C)")

    B, H, W, C = images_uint8.shape
    cols = min(4, B)
    rows = int(np.ceil(B / cols))

    plt.figure(figsize=(cols * 3, rows * 3))
    for i in range(B):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images_uint8[i])
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved image grid to {save_path}")

    plt.show()
    plt.close()


def show_histograms(
    images_uint8: np.ndarray,
    images_01: np.ndarray,
    title: str = "",
    save_path: Path | None = None,
) -> None:
    """Show (and optionally save) pixel histograms before/after normalization.

    Args:
        images_uint8: (B, H, W, C) uint8 images.
        images_01: (B, H, W, C) float images in [0,1].
        title: Plot title prefix.
        save_path: Optional path to save the figure.
    """
    if images_uint8.shape != images_01.shape:
        raise ValueError("images_uint8 and images_01 must have the same shape")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(images_uint8.flatten(), bins=50)
    axes[0].set_title(f"{title} — uint8")
    axes[0].set_xlabel("Pixel value")
    axes[0].set_ylabel("Count")

    axes[1].hist(images_01.flatten(), bins=50)
    axes[1].set_title(f"{title} — normalized [0,1]")
    axes[1].set_xlabel("Pixel value")

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved histogram to {save_path}")

    plt.show()
    plt.close()


def show_heatmap(
    matrix: np.ndarray,
    title: str = "",
    cmap: str = "viridis",
    save_path: Path | None = None,
    show_colorbar: bool = True,
) -> None:
    """Show (and optionally save) a heatmap for a 2D matrix.

    Typical uses:
    - token cosine similarity (N, N)
    - Gram matrices
    - attention score/logit matrices

    Args:
        matrix: (H, W) 2D array.
        title: Plot title.
        cmap: Matplotlib colormap.
        save_path: Optional path to save the figure.
        show_colorbar: Whether to display a colorbar.
    """
    if matrix.ndim != 2:
        raise ValueError("Expected matrix with shape (H, W)")

    plt.figure(figsize=(6, 5))
    im = plt.imshow(matrix, cmap=cmap)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Index")

    if show_colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[viz] Saved heatmap to {save_path}")

    plt.show()
    plt.close()
