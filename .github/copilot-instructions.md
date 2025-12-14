# Copilot Instructions — Computer Vision Basics from Scratch (NumPy-only)

This repo is a learning-first, concept-driven implementation of classical + modern CV using **NumPy only**.
All contributions must maximize intuition: forward-pass math, shape reasoning, and visualization.

---

## Hard Rules (must always follow)
- **NumPy only** for core computation. No torch/tf/jax/keras/autograd.
- Keep examples **toy-scale**: small arrays, tiny images, short videos.
- Prefer **explicit math** over abstractions.
- Every function that touches arrays must:
  - state expected shapes in docstring
  - state dtype assumptions if relevant (prefer float32)
- When implementing attention/softmax/normalization: must be numerically stable.

---

## Allowed Libraries
- numpy
- matplotlib (visualization)
- pillow (PIL) or imageio (optional: basic image I/O only)
- standard library only otherwise (typing, dataclasses, pathlib, math)

## Disallowed
- DL frameworks: torch, tensorflow, jax, keras
- autograd engines
- GPU / CUDA / acceleration tricks
- big training pipelines, datasets, dataloaders

---

## Repo Structure Rules
- Each stage folder:
  - `README.md` (concept notes + checklist + equations + reflection)
  - `src/` (stage code only)
- Shared reusable code goes in `utils/src/`.
- Avoid copy-pasting helpers across stages. Put it into `utils/`.
- Keep files small and focused:
  - Prefer multiple ~100–250 line modules over one 1000-line file.

---

## Code Style Rules
- Python 3.10+.
- Type hints for public functions.
- Docstring template must include:
  - Purpose (1–2 lines)
  - Args/Returns
  - Shapes (explicit)
  - Notes on stability and edge cases

### Shape Notation
Use these consistently:
- `H, W, C` for image dims
- `T` time / frames
- `P` patch size
- `N` number of tokens / patches
- `D` embedding dimension
- `B` batch (only if needed; default avoid batch unless concept requires it)
- `Hh` number of heads (avoid conflict with height)

Example:
- `x: (H, W, C)`
- `patches: (N, P*P*C)`
- `tokens: (N, D)`
- `attn: (Hh, Nq, Nk)`
---
## File Granularity (IMPORTANT)

Follow a concept-driven file structure.

Rules:
- One file per **conceptual unit**, not per function.
- Named models (AutoEncoder, ViT, DETR) must live in their own files.
- Closely related functions should be grouped in the same module.
- Do NOT put implementation logic in `main.py`.

If generating code:
- Place reusable logic in appropriate modules (`models/`, `layers/`, `ops/`).
- Keep `main.py` as an orchestrator only.

---

## Numerical Stability Requirements
- Softmax must be stable:
  - subtract max on last axis
  - handle large negatives for masking
- Normalization (LayerNorm, standardization) must include epsilon (e.g., 1e-5).
- Avoid silent int math: cast to float32 early.

---

## Visualization Rules
Every stage should include at least one “intuition plot”:
- convolution: kernels + feature maps
- DCT: coefficient heatmap + reconstructions
- compression: error images / residual energy
- patch tokens: patch grid visualization
- attention: attention map
- MAE: masked vs reconstructed
- DETR/tracking: association matrix visualization

Plots must have titles/labels and be interpretable.

---

## Testing Expectations (lightweight)
When implementing a core primitive, include at least one:
- shape assertion
- numerical sanity check (e.g., softmax rows sum to 1)
- equivalence test (e.g., conv loop vs conv im2col)

Avoid heavy test frameworks unless needed. Prefer small `if __name__ == "__main__":` demos.

---
## Commit Messages (MUST follow Conventional Commits v1.0.0)
Format:
`<type>(<scope>): <description>`

Types to use in this repo:
feat, fix, docs, refactor, test, chore, perf, style

Scope must be either a stage folder (e.g., 06_self_attention_from_scratch)
or a subsystem (utils, attention, dct, convolution, patchify, mae, detr, tracking, video, viz, repo).

Examples:
- feat(05_patch_tokens_vit): add patchify/unpatchify and token projection
- fix(attention): handle masking with large negative logits safely
- docs(ROADMAP): clarify sparse attention families
- perf(convolution): add im2col vectorized convolution
- chore(repo): add templates and gitignore

---
## PR Title
Use Conventional Commits style:
`feat(scope): ...`, `fix(scope): ...`, `docs(scope): ...`

---

## If uncertain
Choose the option that increases interpretability and aligns with:
signal processing → compression → tokens → attention → MAE → DETR → tracking.
