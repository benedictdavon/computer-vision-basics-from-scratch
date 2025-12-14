import matplotlib.pyplot as plt
import numpy as np


def show_image_grid(images_uint8: np.ndarray, title: str = "") -> None:
    """Show a simple grid of images.

    Args:
        images_uint8: (B, H, W, C) uint8 images.
        title: Plot title.
    """
    B, H, W, C = images_uint8.shape
    cols = min(4, B)
    rows = int(np.ceil(B / cols))

    plt.figure()
    for i in range(B):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images_uint8[i])
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_histograms(
    images_uint8: np.ndarray, images_01: np.ndarray, title: str = ""
) -> None:
    """Show pixel histograms before/after normalization.

    Args:
        images_uint8: (B, H, W, C) uint8 images.
        images_01: (B, H, W, C) float images in [0,1].
    """
    plt.figure()
    plt.hist(images_uint8.flatten(), bins=50)
    plt.title(f"{title} — uint8")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(images_01.flatten(), bins=50)
    plt.title(f"{title} — normalized [0,1]")
    plt.tight_layout()
    plt.show()
