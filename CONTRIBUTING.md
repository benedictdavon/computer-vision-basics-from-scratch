# Style Guide (NumPy CV)

## File Naming
- Use **snake_case** for all files
- Examples: `dct.py`, `attention.py`, `patchify.py`

---

## Function Naming
- Use **verbs + object** to clearly express intent
- Examples:
  - `patchify_image`
  - `compute_attention`
  - `apply_quantization`

---

## Variable Naming & Dimensions

Use **consistent dimension letters** throughout the project:
- `H` — height
- `W` — width
- `C` — channels
- `T` — time / frames
- `P` — patch size
- `N` — number of tokens / patches
- `D` — embedding dimension
- `Hh` — number of attention heads

Avoid ambiguous names like `a`, `b`, `tmp` unless the scope is extremely small.
Prefer explicit names that encode meaning and shape.

---

## Shape Documentation

Always document **key tensor shapes** near transformations.

This can be done via:
- inline comments
- docstrings

Example:

```python
tokens = patches @ W  # (N, P*P*C) @ (P*P*C, D) -> (N, D)
```

Shapes should be readable without needing to run the code.

---

## File Granularity Rules

This project follows a **concept-driven file organization**.

### Core Principle
> **One file per conceptual unit, not per function.**

Avoid both extremes:
- ❌ One giant file containing everything
- ❌ One file per tiny helper function

---

### What Goes in Its Own File?

Create a separate file when the entity is a **named concept**
in the roadmap, literature, or lecture material.

Examples:
- `AutoEncoder` → `models/autoencoder.py`
- `ToyViT` → `models/vit_toy.py`
- `MultiHeadAttention` → `layers/attention.py`
- `Conv2D` implementations → `layers/conv2d.py`

These files may contain:
- one main class
- or a small group of **closely related** functions

---

### What Should NOT Get Its Own File?

Do **not** create a new file for:
- tiny helper functions
- one-line wrappers
- functions that only make sense together

Instead, group them logically within a single module.

Example:

```text
attention.py
- stable_softmax
- scaled_dot_product_attention
- multi_head_attention_forward
```

---

## `main.py` Rule (Strict)

Each stage must contain **exactly one** `main.py`.

`main.py` is responsible for:
- assembling components
- running the forward pass
- printing shape traces
- visualizing results

`main.py` must **NOT**:
- contain core math
- define reusable layers or models
- hide logic that belongs in modules

If math or reusable logic appears in `main.py`,
it should be moved into an appropriate module.

---

## Local vs Shared Code

- Code that is **reusable across stages** → `utils/src/`
- Code that is **stage-specific** → stage `src/`

Do not prematurely move code into `utils/`.
Prefer **clear duplication** over premature abstraction.

---

## Directory Intent

- `models/` → named architectures (AutoEncoder, ViT, DETR, etc.)
- `layers/` → reusable building blocks (conv, linear, attention)
- `ops/` → math utilities (softmax, masking, DCT helpers)
- `viz/` → visualization helpers

---

This style guide exists to keep the codebase:
- readable
- modular
- concept-aligned with the roadmap

If you are unsure whether something deserves its own file,
ask:
> “Is this a named concept I would explain in a lecture or paper?”
