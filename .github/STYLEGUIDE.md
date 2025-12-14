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
