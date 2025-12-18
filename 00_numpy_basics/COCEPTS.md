# Stage 00 — CONCEPTS.md
#
# Project: Computer Vision Basics from Scratch (NumPy-only)
# Stage: 00_numpy_basics
#
# Purpose:
# - Lock in the mental models that will power every later stage.
# - Make tensor reasoning automatic: shapes, axes, layouts, masking, stability.
# - Keep this file framework-agnostic (works for NumPy, PyTorch, JAX, etc.).

---

## 0) How to Use This File

Stage 00 code is your "tensor playground."
This `CONCEPTS.md` is your "tensor compass."

When something breaks later (ViT, attention, MAE, DETR), revisit:
- **Shapes & axes**
- **Broadcasting**
- **Layout**
- **Masking**
- **Numerical stability**

If you can reason correctly about those five, most CV code becomes “just plumbing.”

---

## 1) Tensors, Shapes, and Why Shape > Values

### What is a tensor (in this project)?
A tensor is simply an N-dimensional array with:
- a **shape** (how many axes + size per axis),
- a **dtype** (how numbers are stored),
- a **layout/memory order** (how it sits in memory).

In CV, tensors are **data containers** for images, tokens, features, similarity matrices, etc.

### Common CV shapes you must recognize instantly

**Images**
- Grayscale image: `(H, W)`
- RGB image: `(H, W, C)`
- Batch of RGB images: `(B, H, W, C)` (NHWC)
- Batch in channel-first: `(B, C, H, W)` (NCHW)

**Flattened / feature vectors**
- Flattened image batch: `(B, H*W*C)` (or `(B, D)`)

**Tokens (ViT-style)**
- Tokens: `(B, N, D)`
  - `N` = number of tokens/patches
  - `D` = token embedding dimension

**Similarity / attention**
- Token similarity: `(B, N, N)`
- Attention weights: `(B, Heads, N, N)`
- Attention logits: same shape as weights before softmax

### Why shape matters more than values
Most CV bugs are "correct math on the wrong axis."
Your code can run, produce numbers, and still be totally wrong.

Rule:
> If you cannot explain the shape after every transform, you don’t understand the pipeline yet.

---

## 2) Axes: What do axis=0,1,2 actually mean?

### Axis is “which dimension am I reducing or operating over?”
If `images` is `(B, H, W, C)`:
- `axis=0` refers to **B**
- `axis=1` refers to **H**
- `axis=2` refers to **W**
- `axis=3` refers to **C**

So:
- `images.mean(axis=0)` averages over batches → shape `(H, W, C)`
- `images.mean(axis=(0,1,2))` averages over B,H,W → shape `(C,)`
  - this is exactly "per-channel mean across the dataset/batch"

This is why dataset mean/std is typically computed as:
- mean over all pixels and all images, separately per channel.

### Key idea: reducing axes removes them (unless keepdims=True)
If `images` is `(B,H,W,C)`:
- `mean_c = images.mean(axis=(0,1,2))` → `(C,)`
- `mean_c_keep = images.mean(axis=(0,1,2), keepdims=True)` → `(1,1,1,C)`

Both represent the same values, but the `keepdims=True` version is easier to broadcast safely.

---

## 3) keepdims: The small flag that prevents big bugs

### What keepdims does
`keepdims=True` preserves the reduced axes with size 1 so broadcasting works “by design.”

Example (NHWC):
- `images`: `(B,H,W,C)`
- `mean_c_keep`: `(1,1,1,C)`
Then:
- `images - mean_c_keep` broadcasts cleanly and explicitly.

Without `keepdims`, you get `(C,)`, which also broadcasts in NumPy — but it becomes easier to accidentally broadcast in a wrong place if layout changes.

Rule:
> Use `keepdims=True` when the next operation expects broadcasting.

Especially important when you support both NHWC and NCHW:
- NHWC mean: reduce `(0,1,2)` → keepdims gives `(1,1,1,C)`
- NCHW mean: reduce `(0,2,3)` → keepdims gives `(1,C,1,1)`

---

## 4) Broadcasting: The invisible superpower (and invisible foot-gun)

### Broadcasting is "automatic alignment of size-1 dimensions"
Broadcasting lets you apply per-channel operations without loops.

Example (NHWC):
- `images`: `(B,H,W,C)`
- `mean_c`: `(C,)` or `(1,1,1,C)`
- `images - mean_c` works because NumPy aligns dimensions from the end.

### Why broadcasting is dangerous
Broadcasting can succeed even if you meant to subtract a different axis.

Classic trap:
- You compute a mean incorrectly (wrong axis),
- broadcasting still works,
- results look plausible,
- model fails later.

Rule:
> If broadcasting is involved, always write a comment with the shape transformation.

Example:
- `images_norm = (images - mean_c) / std_c  # (B,H,W,C) - (C,) -> (B,H,W,C)`

---

## 5) Layout: NHWC vs NCHW (and why it matters later)

### What is “layout”?
Layout is the convention for where the channel dimension lives.
- NHWC: `(B,H,W,C)` (common in TensorFlow / many image datasets)
- NCHW: `(B,C,H,W)` (common in PyTorch, often faster on GPUs)

Even in NumPy, you need to understand both because:
- papers and libraries will mention both,
- convolution implementations are easier in one layout depending on your approach,
- attention/tokenization often starts from NHWC images.

### Transpose is not “free”
`transpose` changes the view of memory:
- It can create **non-contiguous arrays**
- Some operations may become slower or require copies later

You don’t optimize for speed here — but you must understand what “contiguous” means conceptually:
- **contiguous**: memory laid out linearly in the same order
- `ensure_contiguous` can be used when needed for downstream ops

Rule:
> Treat layout conversion as a “semantic transform.” Always sanity-check round-trips.

Example sanity check:
- `images == nchw_to_nhwc(nhwc_to_nchw(images))` within tolerance

---

## 6) Dtypes: Why uint8 is a trap

### Why images are uint8
Most raw image datasets store pixels as uint8:
- values in `[0, 255]`
- compact, cheap storage

### Why uint8 breaks math
uint8 arithmetic wraps around.

Example:
- `np.uint8(250) + np.uint8(10)` → wraps (mod 256), not 260

So CV pipelines typically do:
1) `uint8 -> float32`
2) scale to `[0,1]`
3) normalize (mean/std)
4) proceed with linear algebra

Rule:
> Never do meaningful math in uint8.

### Why float32 is the default in deep learning
- good balance of precision and speed/memory
- stable enough for typical pipelines with eps tricks
- float64 is fine for analysis, but you’ll learn “float32 habits” that map to real training later

---

## 7) Normalization: What it really does

### Scaling vs normalization
- Scaling: `x / 255.0` maps `[0,255] -> [0,1]`
- Normalization: `(x - mean) / std` re-centers and re-scales

### Why normalization matters
Many operations assume “inputs are roughly standardized”:
- dot-products become meaningful
- cosine similarity becomes stable
- optimization (later) becomes easier
- attention logits are less extreme

In attention, raw dot-products can get huge:
- softmax becomes peaky
- gradients (in real training) vanish/explode

Even though Stage 00 doesn’t train, you build the intuition:
> normalization is not a ritual — it controls numeric scale.

---

## 8) Vectorization vs Loops: Math as array operations

### What vectorization means here
Vectorization means expressing computation as:
- whole-array operations
- reductions over axes
- broadcasting
- matrix multiplications

Instead of explicit Python loops.

### Why we still write loop versions
Loop versions are:
- correctness references
- easier to understand step-by-step
- good for sanity checking vectorized implementations

Rule:
> Keep both versions only if the loop version teaches something.

For example:
- per-image mean: loop vs vectorized
- gamma correction: nested loops vs `np.power`
- pairwise distances: double loop vs matrix trick

### Pairwise distance trick (core intuition)
For `x: (N,D)`, `y: (M,D)`:
- squared distance:
  - `||x_i - y_j||^2 = ||x_i||^2 - 2 x_i·y_j + ||y_j||^2`
This identity is the backbone of:
- kNN / retrieval
- clustering
- similarity search
- attention-like score computations

---

## 9) Numerical stability: “works on small numbers” is a trap

### Where instability comes from
Common sources:
- dividing by small numbers → inf
- taking log of 0 → -inf
- exp of large numbers → inf
- subtracting large similar numbers → precision loss

Stage 00 stability patterns:
- eps in denominators
- `stable_softmax`
- NaN/Inf checks

### Stable softmax (the central trick)
Softmax:
- `softmax(x_i) = exp(x_i) / sum_j exp(x_j)`

If `x` contains large values, `exp(x)` overflows.

Trick:
- subtract max before exp:
  - `exp(x - max(x))` is safe because the largest term becomes `exp(0)=1`

This is not optional in attention.
Attention logits can easily be large because they come from dot-products.

Rule:
> Always use stable softmax, always.

### Attention masks use “-inf” (or a large negative)
Masking in attention typically adds a bias:
- valid positions: add 0
- invalid positions: add `-inf` (or `-1e9`)

Then softmax:
- `exp(-1e9) ≈ 0`
So masked entries contribute nothing.

This is the conceptual bridge to transformers.

---

## 10) Masking & indexing: the shape-shifting operations

Masking is everywhere in modern CV:
- MAE masks patches
- attention masks tokens
- detection masks padded regions
- tracking masks invalid detections

### Boolean mask vs index array
- boolean mask: shape `(N,)`, values True/False
- index array: shape `(K,)`, values are integer indices

Both are useful. You should be comfortable converting between them:
- `indices = np.where(mask)[0]`
- `mask = make_boolean_mask(N, indices)`

### Gather / restore (MAE mental model)
MAE style:
1) choose masked indices
2) gather visible tokens
3) run encoder on visible tokens
4) restore full sequence with mask tokens inserted

Key concept:
> masking changes the effective sequence length, which changes compute.

That is the entire idea behind masked modeling efficiency.

---

## 11) Linear algebra building blocks: CV flavor

### Dot product as similarity
If `u` and `v` are vectors:
- `u · v` is large if they align and have large magnitude

In attention, `Q @ K.T` is literally “all pair dot products.”

### Gram matrix
If `x: (N,D)`:
- `gram = x @ x.T` → `(N,N)` with similarity between all pairs of tokens

You can view this as:
- a similarity graph adjacency matrix
- the raw input to attention (before scaling/softmax)

### Cosine similarity vs dot product
Cosine similarity removes magnitude:
- `cos(u,v) = (u·v) / (||u|| ||v||)`

Why it matters:
- dot product cares about scale (magnitude)
- cosine similarity cares about direction (pattern)

In attention:
- dot-product attention is scale-sensitive
- normalization tricks (like scaling by sqrt(D)) exist to control this

Stage 00 takeaway:
> similarity can be defined in many ways; attention uses dot-product + scaling + softmax.

### L2 norm
`||x||_2 = sqrt(sum x_i^2)`
Used for:
- normalization
- cosine similarity
- stability checks
- measuring “energy” of signals

---

## 12) PCA & SVD: Structure discovery, not magic

### What PCA is (conceptually)
PCA finds directions of maximum variance:
- reduces dimensionality
- reveals dominant patterns in data

In CV, PCA intuition helps you understand:
- embeddings live in a space with structure
- some directions matter more
- compression ideas (later) are related to “keep important components”

### Why SVD appears everywhere
SVD decomposes a matrix into:
- directions + strengths

This is connected to:
- low-rank approximations
- compression
- redundancy reduction
- efficient representations

Stage 00 use:
- keep it tiny and conceptual
- see that “data lives on a lower-dimensional structure”

---

## 13) Connecting Stage 00 to Modern Models

### ViT / patch tokens
- reshape from `(B,H,W,C)` → `(B,N,D)`
- every token is just a vector
- everything becomes linear algebra after tokenization

### Attention
- similarity matrix `(B,N,N)` from dot products
- softmax along last axis
- masking by adding negative infinity bias

### MAE
- random masking indices
- gather visible tokens
- restore with mask tokens
- compare reconstructions later

### DETR-style thinking
- everything is a set of vectors
- similarity / matching uses matrix operations
- masking matters for padded variable-length sets

### Video later
- add time dimension `T`:
  - images become `(B,T,H,W,C)`
  - tokens become `(B,T,N,D)` or `(B,N,T,D)` depending on design
Stage 00 axis reasoning becomes essential.

---

## 14) Debugging Rules (Non-Negotiable)

1) **Trace shapes after every semantic transform**
   - layout change
   - normalization
   - reshape/tokenization
   - masking/gather/restore
   - similarity matrix creation

2) **Validate with sanity checks**
   - loop vs vectorized equality
   - NHWC->NCHW->NHWC round-trip
   - softmax sums to 1
   - no NaNs/Infs

3) **Write shape comments where humans get confused**
   - especially around `(B,N,D)` and `(N,N)`

4) **Never assume broadcasting is correct**
   - prove it with keepdims and comments

5) **Make numeric scale visible**
   - min/max/mean traces
   - histograms
   - heatmaps for similarity matrices

---


## 15) Checklist: “Do I really understand Stage 00?”

You understand Stage 00 if you can explain, without guessing:
- why `axis=(0,1,2)` yields channel stats in NHWC
- why keepdims reduces broadcasting bugs
- why uint8 overflows silently
- why stable softmax subtracts max
- what `X @ X.T` means geometrically
- how masking changes shapes and compute
- how to go from images to tokens to similarity matrices

If any of these feel fuzzy, revisit the demos and re-trace shapes.
