# Style Guide (NumPy CV)

## File naming
- snake_case: `dct.py`, `attention.py`, `patchify.py`

## Function naming
- verbs + object: `patchify_image`, `compute_attention`, `apply_quantization`

## Variables
Use consistent dimension letters:
- `H, W, C, T, P, N, D, Hh`
Avoid ambiguous names like `a`, `b`, `tmp` unless tiny.

## Shapes
Always comment or docstring key shapes near transformations.
Example:
`tokens = patches @ W  # (N, P*P*C) @ (P*P*C, D) -> (N, D)`

## File Granularity Rules

This project follows a **concept-driven file organization**.

### Core Principle
> **One file per conceptual unit, not per function.**

Avoid both extremes:
- ❌ One giant file containing everything
- ❌ One file per tiny helper function

---

### What goes in its own file?

Create a separate file when the entity is a **named concept** in the roadmap or literature.

Examples:
- `AutoEncoder` → `models/autoencoder.py`
- `ToyViT` → `models/vit_toy.py`
- `MultiHeadAttention` → `layers/attention.py`
- `Conv2D` implementations → `layers/conv2d.py`

These files may contain:
- one main class
- or a small group of closely related functions

---

### What should NOT get its own file?

Do **not** create a new file for:
- tiny helper functions
- one-line wrappers
- functions that only make sense together

Instead, group them logically.

Example:
```text
attention.py
- stable_softmax
- scaled_dot_product_attention
- multi_head_attention_forward
