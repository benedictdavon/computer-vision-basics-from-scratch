from __future__ import annotations

import numpy as np


def random_mask_indices(
    num_tokens: int,
    mask_ratio: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random mask indices (sorted) for masking tokens.

    Args:
        num_tokens: Total tokens N.
        mask_ratio: Fraction to mask in [0, 1].
        seed: Optional seed (does not touch global RNG).

    Returns:
        mask_idx: (num_mask,) int64 sorted indices, True=masked convention is applied later.
    """
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError("mask_ratio must be in [0, 1]")

    rng = np.random.default_rng(seed)
    num_mask = int(round(num_tokens * mask_ratio))
    num_mask = max(0, min(num_mask, num_tokens))

    perm = rng.permutation(num_tokens)
    mask_idx = np.sort(perm[:num_mask])
    return mask_idx


def make_boolean_mask(num_tokens: int, mask_idx: np.ndarray) -> np.ndarray:
    """Create boolean mask of shape (N,) where True means masked."""
    if mask_idx.ndim != 1:
        raise ValueError("mask_idx must be 1D")
    if mask_idx.size > 0 and (mask_idx.min() < 0 or mask_idx.max() >= num_tokens):
        raise ValueError("mask_idx contains out-of-range indices")

    mask = np.zeros((num_tokens,), dtype=bool)
    mask[mask_idx] = True
    return mask


def apply_token_mask(
    tokens: np.ndarray,
    mask: np.ndarray,
    mask_token: np.ndarray,
) -> np.ndarray:
    """Replace masked tokens with mask_token.

    Supports:
    - tokens: (N, D) with mask: (N,)
    - tokens: (B, N, D) with mask: (N,) or (B, N)

    Conventions:
    - mask == True means "masked".
    """
    if mask_token.ndim != 1:
        raise ValueError("mask_token must have shape (D,)")
    d = mask_token.shape[0]

    if tokens.ndim == 2:
        n, d_tokens = tokens.shape
        if d_tokens != d:
            raise ValueError("mask_token dim D must match tokens.shape[1]")
        if mask.shape != (n,):
            raise ValueError("For (N,D) tokens, mask must have shape (N,)")

        out = tokens.copy()
        out[mask] = mask_token
        return out

    if tokens.ndim == 3:
        b, n, d_tokens = tokens.shape
        if d_tokens != d:
            raise ValueError("mask_token dim D must match tokens.shape[2]")

        if mask.shape == (n,):
            mask_b = np.broadcast_to(mask[None, :], (b, n))
        elif mask.shape == (b, n):
            mask_b = mask
        else:
            raise ValueError("For (B,N,D) tokens, mask must be (N,) or (B,N)")

        out = tokens.copy()
        out[mask_b] = mask_token
        return out

    raise ValueError("tokens must have shape (N,D) or (B,N,D)")


def gather_visible_tokens(
    tokens: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather visible tokens (unmasked) for a shared mask.

    This Stage-00 version assumes a single mask shared across batch:
    - tokens: (B, N, D)
    - mask: (N,) where True=masked

    Returns:
        visible_tokens: (B, N_vis, D)
        visible_idx: (N_vis,)
    """
    if tokens.ndim != 3:
        raise ValueError("Expected tokens with shape (B, N, D)")
    b, n, _d = tokens.shape
    if mask.shape != (n,):
        raise ValueError("Expected shared mask with shape (N,)")

    visible_idx = np.where(~mask)[0]
    visible_tokens = tokens[:, visible_idx, :]
    return visible_tokens, visible_idx


def restore_tokens(
    visible_tokens: np.ndarray,
    mask: np.ndarray,
    mask_token: np.ndarray,
) -> np.ndarray:
    """Restore full (B,N,D) tokens from visible tokens using a shared mask.

    Args:
        visible_tokens: (B, N_vis, D)
        mask: (N,) where True=masked
        mask_token: (D,)

    Returns:
        restored: (B, N, D)
    """
    if visible_tokens.ndim != 3:
        raise ValueError("Expected visible_tokens with shape (B, N_vis, D)")
    if mask_token.ndim != 1:
        raise ValueError("Expected mask_token with shape (D,)")

    b, n_vis, d = visible_tokens.shape
    if mask_token.shape[0] != d:
        raise ValueError("mask_token dim D must match visible_tokens.shape[2]")

    n = mask.shape[0]
    visible_idx = np.where(~mask)[0]
    if visible_idx.shape[0] != n_vis:
        raise ValueError("visible_tokens N_vis must match number of ~mask positions")

    restored = np.empty((b, n, d), dtype=visible_tokens.dtype)
    restored[:, mask, :] = mask_token  # broadcast (D,) -> (B, num_mask, D)
    restored[:, visible_idx, :] = visible_tokens
    return restored


def make_attention_mask(valid: np.ndarray, neg_inf: float = -1e9) -> np.ndarray:
    """Create additive attention bias mask.

    Args:
        valid: (B, N) boolean where True means token is valid/attendable.
        neg_inf: large negative for masked positions.

    Returns:
        attn_bias: (B, 1, 1, N) float32 to add to attention logits.
    """
    if valid.ndim != 2:
        raise ValueError("Expected valid with shape (B, N)")

    attn_bias = np.where(valid, 0.0, float(neg_inf)).astype(np.float32)
    return attn_bias[:, None, None, :]
