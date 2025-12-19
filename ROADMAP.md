# Computer Vision Basics from Scratch (NumPy-only)
### A Concept-Driven Learning Roadmap

This repository is a **personal learning project** designed to build deep intuition for modern Computer Vision by **manually implementing core ideas using NumPy only**.

The focus is **understanding**, not performance.

---

## Goals
- Build strong intuition for how CV systems work internally
- Understand classical CV → deep learning → transformers as a single continuum
- Connect:
  - signal processing
  - redundancy & compression
  - representation learning
  - attention & transformers
- Be able to **read CV papers with confidence**
- Reason clearly about **ViT, MAE, DETR, tracking, and video models**

---

## Constraints
- **NumPy only**
- No PyTorch / TensorFlow / JAX
- No autograd
- No GPU tricks or large-scale training
- Focus on:
  - forward pass
  - math
  - matrix operations
  - data flow
- Small images and toy examples are encouraged

---

## How to Use This Roadmap
Each stage contains:
- **Concept**
- **What to implement** (checklist)
- **Intuition built**
- **Connection to modern models**

Each stage should end with:
- A tiny demo (2–16 images or a short video clip)
- At least one visualization
- A short written reflection

**Repo rule (recommended):**
- `main.py` is an orchestrator only.
- Core logic lives in `ops/`, `data/`, `viz/`, `layers/`, `models/`.

---

# Stage 00 — Tensor Semantics, NumPy Ops, and “Vision-Ready” Math

## Concept: Tensors are the interface between images, tokens, and attention

### Implement

# Stage 00 — Tensor Semantics, NumPy Ops, and “Vision-Ready” Math

## Concept: Tensors are the interface between images, tokens, and attention

### Implement

- [x] Tensor tracing utilities (shape/dtype/min/max/mean after each transform)
- [x] Layout conventions + transforms
  - [x] NHWC ↔ NCHW conversion
  - [x] Contiguity checks (`is_contiguous`, `ensure_contiguous`)
  - [x] Channel-wise ops supporting both layouts (mean/std/centering)
- [x] Dtype & normalization safety
  - [x] `uint8 -> float32` conversion (`to_float01`)
  - [x] Per-channel normalization using dataset stats
  - [x] Demonstrate overflow pitfalls + clipping patterns (Stage 00 demo)
- [x] Vectorization vs loops (same output, different computation)
  - [x] Per-image mean (loop vs vectorized)
  - [x] Per-image variance (loop vs vectorized)
  - [x] Per-pixel transform demo (gamma correction)
  - [x] Pairwise squared distances on token-like arrays `(N, D)`
- [x] Numerical stability primitives (future attention prep)
  - [x] `stable_softmax(x, axis=...)`
  - [x] `safe_divide`, `safe_l2_normalize`
  - [x] Checks: sums-to-1, no NaN/Inf
- [x] Linear algebra building blocks (future ViT/MAE/DETR prep)
  - [x] L2 norm helper
  - [x] Gram matrix (`X @ X.T`) on tokens
  - [x] Cosine similarity supporting vector + matrix cases
  - [x] PCA toy via SVD (center → SVD → project)
- [x] Masking primitives (future MAE prep)
  - [x] Random mask indices + boolean mask creation
  - [x] Apply mask token to `(N, D)` tokens
  - [x] Gather visible tokens + restore tokens
  - [x] Attention-style mask / bias construction
- [x] CIFAR-10 mini-loader (NumPy-only)
  - [x] Download + extract
  - [x] Batch iteration (streaming) for dataset mean/std
  - [x] Load a small batch for demos
- [x] Visualization basics
  - [x] Image grid + histograms
  - [x] Save plots to disk for README embedding

### Intuition
- Images aren’t “pictures”; they are **tensors with conventions**.
- Most “model bugs” are actually **shape/layout/dtype** bugs.
- Attention and transformers are “just matrix ops” once tensor semantics are solid.

### Modern Connection
- **ViT**: `(B,H,W,C)` → patches → `(B,N,D)` depends on correct tensor reasoning.
- **MAE**: masking/gather/restore are core mechanics.
- **DETR**: similarity matrices + stable softmax + consistent layout assumptions.

---

# Stage 01 — Classical CV as Signal Processing

## Concept A: Images are signals + linear transforms (representation matters)

### Implement
- [ ] Image tensor formats: `(H, W)`, `(H, W, C)`, `(B, H, W, C)`, `(T, H, W, C)`
- [ ] RGB ↔ grayscale conversion (luma-style)
- [ ] Simple YCbCr-like color transform (per-pixel matrix multiply)
- [ ] (Optional) Chroma subsampling intuition (toy 4:2:0-style downsample/upsample)
- [ ] Pixel statistics & histograms (reuse Stage 00 ops)
- [ ] (Optional but useful) resizing / interpolation basics
  - [ ] nearest-neighbor and bilinear resize
  - [ ] aliasing demo with downsampling

### Intuition
- Representation transforms are **linear maps** on signals.
- Statistics are part of the “data contract” for every pipeline.
- Sampling/interpolation choices change what information survives.

### Modern Connection
- Preprocessing and normalization assumptions are baked into ViT/MAE/DETR.
- Tokenization and patch embeddings inherit these representation choices.

---

## Concept B: Convolution as feature extraction (structured linear layer)

### Implement
- [ ] Naive 2D convolution (loops) for grayscale
- [ ] Multi-channel convolution (RGB input, multiple filters)
- [ ] Vectorized convolution (im2col + GEMM)
- [ ] Blur, Sobel, Laplacian kernels
- [ ] Edge magnitude & orientation
- [ ] Separable convolution (show concept + small speed intuition)

### Intuition
- Convolution is **local dot-product scanning**.
- It encodes locality and translation bias as an inductive prior.

### Modern Connection
- CNNs learn these filters; ViTs often learn similar early patterns.
- Deformable attention ≈ adaptive/local sampling reminiscent of learned convolution.

---

## Concept C: Frequency domain & energy compaction

### Implement
- [ ] 1D DCT-II
- [ ] 2D DCT via separability
- [ ] Energy visualization of DCT coefficients
- [ ] Top-k coefficient reconstruction
- [ ] DCT vs FFT comparison (interpretation + energy compaction)

### Intuition
- Natural images concentrate energy in low frequencies → compression works.
- “Information” is not uniformly distributed in frequency space.

### Modern Connection
- Patch embeddings often behave like learned bases (sometimes frequency-like).
- MAE reconstruction mirrors recovering missing structure from priors.

---

# Stage 02 — Image Compression Pipeline (JPEG-style)

## Concept A: JPEG-like spatial coding (block transform + quantize)

### Implement
- [ ] 8×8 block splitting & merging
- [ ] Block-wise DCT
- [ ] Quantization & dequantization
- [ ] Zigzag scan
- [ ] Run-length encoding (RLE)
- [ ] (Optional) toy “quality factor” to scale quant tables

### Intuition
- Quantization = controlled information loss.
- Many coefficients are small → easy to compress.

### Modern Connection
- Token pruning/masking are “learned compression” analogs.
- MAE masking is omission-based compression.

---

## Concept B: Statistical redundancy & entropy

### Implement
- [ ] Histogram-based probability estimation
- [ ] Shannon entropy calculation
- [ ] Huffman coding (toy)
- [ ] Entropy before/after quantization
- [ ] (Optional) compare RLE-only vs Huffman-only vs both

### Intuition
- Compression = decorrelate → quantize → encode.
- Redundancy becomes shorter codes.

### Modern Connection
- Token distributions and sparsity are statistical structure.
- Attention can be viewed as routing information to reduce uncertainty.

---

# Stage 03 — Video Basics & Temporal Redundancy

## Concept: Motion-based prediction (what changes vs what stays)

### Implement
- [ ] Frame differencing
- [ ] Block matching (SSD / SAD)
- [ ] Motion compensation
- [ ] Residual visualization
- [ ] Toy GOP (I / P / B frame simulation)
- [ ] (Optional) simple stabilization / global motion (toy affine) demo

### Intuition
- Most change is explainable by motion; residuals capture what motion can’t.
- Temporal redundancy is the key to video efficiency.

### Modern Connection
- Tracking = motion + identity.
- Video transformers handle dependencies beyond motion (long-range, semantics).

---

# Stage 04 — Patch Tokens (Bridge to Transformers)

## Concept: Images as sequences (tokenization is representation)

### Implement
- [ ] Patch extraction (non-overlapping)
- [ ] (Optional) overlapping patches
- [ ] Patch flattening & linear projection
- [ ] 2D positional encodings (sin/cos or learned values)
- [ ] Unpatchify (reconstruct image grid for sanity)

### Intuition
- ViT treats images like sequences; position must be injected explicitly.
- Tokenization is not “just reshaping” — it is a representational choice.

### Modern Connection
- Core ViT / MAE input representation.

---

# Stage 05 — Self-Attention from Scratch

## Concept: Attention as content-addressable memory

### Implement
- [ ] Q, K, V projections
- [ ] Scaled dot-product attention
- [ ] Softmax (numerically stable)
- [ ] Multi-head attention
- [ ] Attention masking (padding/valid tokens, causal if desired)
- [ ] LayerNorm (manual)
- [ ] Residual connections
- [ ] (Recommended) MLP block (2-layer) + activation (GELU or ReLU)
- [ ] (Optional) simple “Transformer block” forward pass

### Intuition
- Queries ask, keys match, values retrieve.
- Attention builds dynamic graphs between tokens.

### Modern Connection
- Core of ViT, MAE, DETR, video transformers.
- Sparse attention modifies which keys are visible.

---

# Stage 06 — Masked Modeling (MAE Intuition)

## Concept: Learning by reconstruction (mask → encode → decode)

### Implement
- [ ] Random patch masking (e.g., 75%)
- [ ] Encoder on visible patches only
- [ ] Lightweight decoder for reconstruction (toy)
- [ ] Pixel-space or DCT-space loss (forward computation)
- [ ] Reconstruction visualization
- [ ] (Optional) tiny manual training loop for a linear decoder (few steps, educational)

### Intuition
- Masking forces semantic structure learning.
- Encoder learns structure; decoder fills in detail.

### Modern Connection
- MAE / BERT-style pretraining and encoder–decoder asymmetry.

---

# Stage 07 — Sparse Attention Families

## Concept A: Windowed attention

### Implement
- [ ] Partition tokens into windows
- [ ] Attention within each window
- [ ] Compare with global attention maps

### Intuition
- Locality bias without convolution.

### Modern Connection
- Swin-style transformers.

---

## Concept B: Pyramid / hierarchical tokens

### Implement
- [ ] Token downsampling / merging
- [ ] Multi-scale token sets
- [ ] Cross-scale attention

### Intuition
- Objects exist at multiple scales.

### Modern Connection
- Pyramid ViTs for dense prediction.

---

## Concept C: Deformable attention intuition

### Implement
- [ ] Offset-based key sampling (toy offsets)
- [ ] Attention over sampled keys only
- [ ] Sampling visualization

### Intuition
- Attend where it matters, not everywhere.

### Modern Connection
- Deformable attention for detection & tracking.

---

# Stage 08 — DETR-Style Detection

## Concept: Detection as set prediction

### Implement
- [ ] Image tokens from patches
- [ ] Learnable object queries
- [ ] Cross-attention (queries → image tokens)
- [ ] Linear heads for class & box
- [ ] Hungarian matching (NumPy)
- [ ] Toy IoU + L1 losses (forward computation)
- [ ] Visualization: predicted boxes on toy images

### Intuition
- Object queries are competing explanations.
- Matching resolves permutation ambiguity.

### Modern Connection
- DETR, object queries, transformer decoders.

---

# Stage 09 — Tracking & Video Memory

## Concept A: Tracking by attention / similarity

### Implement
- [ ] Cross-frame similarity (tokens or object features)
- [ ] Association score matrix
- [ ] Greedy or Hungarian matching
- [ ] (Optional) add motion prior: simple constant-velocity or box propagation

### Intuition
- Tracking = correspondence over time under ambiguity.

### Modern Connection
- Transformer-based tracking uses attention for association.

---

## Concept B: Temporal memory

### Implement
- [ ] Memory token bank
- [ ] Current frame attends to memory
- [ ] Memory update strategies (FIFO, top-k, stride)
- [ ] Visualization: what memory is kept vs dropped

### Intuition
- Long-term reasoning requires memory compression.
- Memory is a budgeted resource, not “store everything”.

### Modern Connection
- Video transformers with long-term memory.

---

## Final Outcome
After completing this roadmap, you should be able to:
- Read ViT / MAE / DETR papers comfortably
- Reason about attention, masking, sparsity, and compression
- Explain modern CV models from first principles
- See classical CV and deep learning as one coherent system

---

**Project philosophy**
> If you can implement it with NumPy, you truly understand it.
