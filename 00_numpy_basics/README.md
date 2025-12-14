# Stage 00 — NumPy Basics & Tensor Intuition

## Goal
Build intuition for:
- arrays as tensors
- shapes and broadcasting
- basic linear algebra used throughout CV

This stage ensures all later stages have a shared mental model.

---

## What You Will Create in This Stage

By the end of Stage 00, you should have:

### 1) A tiny “tensor lab” module (core utilities)
- A small set of scripts that demonstrate:
  - shape manipulation
  - broadcasting
  - vectorization patterns
  - dtype pitfalls
  - sanity-check helpers for shapes

### 2) A reproducible mini dataset loader (tiny + real)
- A lightweight way to fetch/load a small real dataset (no deep learning frameworks).
- The dataset is only used to provide real-looking arrays for your demos.

### 3) One stage `main.py` mini project
- A runnable script that ties everything together:
  - loads a small dataset
  - performs a sequence of tensor transformations
  - prints shape traces
  - produces a few simple visualizations

---

## Concepts Covered
- Array shapes and axes
- Broadcasting rules
- Vectorization vs loops
- Dtype and numerical precision
- Basic linear algebra operations used throughout CV

---

## Implementation Checklist

### A) Tensor & Shape Fundamentals
- [ ] Create arrays with explicit shapes (`(H, W)`, `(H, W, C)`, `(N, D)`)
- [ ] Practice indexing and slicing across axes
- [ ] Use `reshape`, `transpose`, `swapaxes`, and `expand_dims` intentionally
- [ ] Implement a small helper `assert_shape(x, expected)` for sanity checks

### B) Broadcasting (Must Master)
- [ ] Demonstrate scalar + vector + matrix broadcasting with clear examples
- [ ] Show common broadcasting bugs (wrong axis) and how to fix with `reshape`
- [ ] Implement examples like:
  - per-channel normalization
  - per-row / per-column operations

### C) Vectorization vs Loops
- [ ] Implement loop vs vectorized versions for:
  - mean/variance
  - per-pixel transforms
  - pairwise distances (small scale)
- [ ] Verify equality between implementations (within tolerance)
- [ ] (Optional) Quick timing comparisons on small arrays (educational only)

### D) Dtype & Numerical Precision
- [ ] Demonstrate uint8 overflow pitfalls
- [ ] Convert to float safely (`astype(np.float32)`)
- [ ] Show normalization and clipping patterns
- [ ] Explain why float32 is the default for later stages

### E) Linear Algebra Building Blocks (CV-relevant)
- [ ] Dot products and matrix multiplication: `@`
- [ ] Vector norms and cosine similarity
- [ ] Outer products and rank-1 structure
- [ ] Simple PCA intuition with SVD (conceptual; optional implementation)

---

## Mini Project — “Tensor Playground on Real Images”

**Purpose:** Use a small, real dataset to repeatedly practice the tensor operations
you will need later for:
- patchify / tokenization
- attention shapes
- reconstruction / residuals
- detection & tracking tensor layouts

### Dataset Choice (Pick One)

**Option 1 (Recommended): CIFAR-10**
- Small real RGB images: `(32, 32, 3)`
- Easy to download and handle
- Perfect size for toy experiments

**Option 2: MNIST**
- Smaller grayscale images: `(28, 28)`
- Even simpler, but less representative for RGB pipelines

> You do NOT need labels for Stage 00; this is purely for tensor practice.

---

## Mini Project Outputs (What `main.py` should do)

Your `main.py` should:

### 1) Load a batch of images
- Load `B` images as a tensor:
  - `images: (B, H, W, C)` for CIFAR-10
  - or `images: (B, H, W)` for MNIST

### 2) Print “shape traces”
- After every transformation, print:
  - tensor name
  - shape
  - dtype
  - min/max

Example:
- `images -> (B, 32, 32, 3), uint8, min=0, max=255`

### 3) Demonstrate core transformations (the “playground”)
- Convert dtype to float32
- Normalize:
  - per-image normalization
  - per-channel normalization
- Reshape examples:
  - flatten image to vector: `(B, H*W*C)`
  - convert to tokens style: `(B, N, D)` (toy, just reshape)
- Broadcasting examples:
  - subtract per-channel mean (broadcast across H,W)

### 4) Create visualizations (minimum set)
- [ ] Show a grid of original images
- [ ] Show the same grid after normalization
- [ ] Show a histogram of pixel values before/after normalization

### 5) Validate correctness with small checks
- [ ] Assert expected shapes after each step
- [ ] Verify loop vs vectorized results match (within tolerance)

---

## Suggested Files to Create (Stage 00)

Keep this stage simple but structured:

- `src/main.py`
- `src/ops/tensor_basics.py`
  - shape helpers, reshape patterns, broadcasting demos
- `src/ops/vectorization.py`
  - loop vs vectorized implementations + checks
- `src/ops/dtype_and_norm.py`
  - uint8 pitfalls, normalization helpers
- `src/data/cifar10.py` (or `mnist.py`)
  - tiny downloader/loader (NumPy-only usage)
- `src/viz/plots.py`
  - image grids, histograms

---

## Key Notes
- Shapes matter more than values
- Prefer explicit reshaping over implicit magic
- Always reason about `(H, W, C)` vs `(N, D)`
- If you can’t explain a broadcasting operation, rewrite it explicitly

---

## Reflection (Write this after completing Stage 00)
Answer in your own words:
- What did I misunderstand about shapes before this stage?
- Which broadcasting bug surprised me the most?
- What shape transformations show up everywhere in later CV pipelines?
- What conventions will I enforce in future stages to avoid confusion?
