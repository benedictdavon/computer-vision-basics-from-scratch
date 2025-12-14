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
- A tiny demo (2–5 images or short video)
- At least one visualization
- A short written reflection

---

# Stage 0 — Core Math & Image Representation

## Concept: Images are tensors + linear operators

### Implement
- [ ] Image tensor formats: `(H, W)`, `(H, W, C)`, `(T, H, W, C)`
- [ ] RGB ↔ grayscale conversion
- [ ] Simple YCbCr-like color transform
- [ ] Normalization (uint8 ↔ float32)
- [ ] Pixel statistics & histograms

### Intuition
- Images are **signals**, not pictures
- Everything later is tensor transformation + linear algebra

### Modern Connection
- All ViT / MAE / DETR pipelines assume correct tensor semantics
- Video models rely on consistent spatiotemporal representation

---

# Stage 1 — Classical CV as Signal Processing

## Concept: Convolution as feature extraction

### Implement
- [ ] Naive 2D convolution (loops)
- [ ] Vectorized convolution (im2col)
- [ ] Blur, Sobel, Laplacian kernels
- [ ] Edge magnitude & orientation
- [ ] Separable convolution

### Intuition
- Convolution = structured linear layer
- Local receptive fields encode inductive bias

### Modern Connection
- CNNs learn filters; ViT removes locality but still learns similar early patterns
- Deformable attention ≈ adaptive convolution

---

## Concept: Frequency domain & energy compaction

### Implement
- [ ] 1D DCT-II
- [ ] 2D DCT via separability
- [ ] Energy visualization of DCT coefficients
- [ ] Top-k coefficient reconstruction
- [ ] DCT vs FFT comparison

### Intuition
- Natural images concentrate energy in low frequencies
- This is the mathematical reason compression works

### Modern Connection
- Patch embeddings implicitly learn frequency-like bases
- MAE reconstruction mirrors frequency recovery

---

# Stage 2 — Image Compression Pipeline

## Concept: JPEG-style spatial coding

### Implement
- [ ] 8×8 block splitting & merging
- [ ] Block-wise DCT
- [ ] Quantization & dequantization
- [ ] Zigzag scan
- [ ] Run-length encoding (RLE)

### Intuition
- Quantization = controlled information loss
- Most information is redundant

### Modern Connection
- Token pruning & masking = learned compression
- MAE masking is omission-based compression

---

## Concept: Statistical redundancy & entropy

### Implement
- [ ] Histogram-based probability estimation
- [ ] Shannon entropy calculation
- [ ] Huffman coding (toy)
- [ ] Entropy before/after quantization

### Intuition
- Compression = decorrelate → quantize → encode
- Redundancy becomes shorter codes

### Modern Connection
- Token distributions & sparsity
- Attention reduces uncertainty by routing information

---

# Stage 3 — Video Basics & Temporal Redundancy

## Concept: Motion-based prediction

### Implement
- [ ] Frame differencing
- [ ] Block matching (SSD / SAD)
- [ ] Motion compensation
- [ ] Residual visualization
- [ ] Toy GOP (I / P / B frame simulation)

### Intuition
- Most video change is explainable by motion
- Residuals capture what motion cannot

### Modern Connection
- Tracking = motion + identity
- Video transformers handle long-range dependencies beyond motion

---

# Stage 4 — Patch Tokens (Bridge to Transformers)

## Concept: Images as sequences

### Implement
- [ ] Patch extraction (non-overlapping)
- [ ] Optional overlapping patches
- [ ] Patch flattening & linear projection
- [ ] 2D positional encodings (sin/cos or learned)

### Intuition
- ViT treats images like sentences
- Position is not implicit — it must be injected

### Modern Connection
- Core ViT / MAE input representation

---

# Stage 5 — Self-Attention from Scratch

## Concept: Attention as content-addressable memory

### Implement
- [ ] Q, K, V projections
- [ ] Scaled dot-product attention
- [ ] Softmax (numerically stable)
- [ ] Multi-head attention
- [ ] Attention masking
- [ ] LayerNorm (manual)
- [ ] Residual connections

### Intuition
- Queries ask, keys match, values retrieve
- Attention builds dynamic graphs between tokens

### Modern Connection
- Core of ViT, MAE, DETR, video transformers
- Sparse attention modifies which keys are visible

---

# Stage 6 — Masked Modeling (MAE Intuition)

## Concept: Learning by reconstruction

### Implement
- [ ] Random patch masking (e.g., 75%)
- [ ] Encoder on visible patches only
- [ ] Lightweight decoder for reconstruction
- [ ] Pixel-space or DCT-space loss
- [ ] Reconstruction visualization

### Intuition
- Masking forces semantic understanding
- Encoder learns structure, decoder fills details

### Modern Connection
- MAE / BERT-style pretraining
- Encoder–decoder asymmetry

---

# Stage 7 — Sparse Attention Families

## Concept A: Windowed attention

### Implement
- [ ] Partition tokens into windows
- [ ] Attention within each window
- [ ] Compare with global attention maps

### Intuition
- Locality bias without convolution

### Modern Connection
- Swin-style transformers

---

## Concept B: Pyramid / hierarchical tokens

### Implement
- [ ] Token downsampling / merging
- [ ] Multi-scale token sets
- [ ] Cross-scale attention

### Intuition
- Objects exist at multiple scales

### Modern Connection
- Pyramid ViTs for dense prediction

---

## Concept C: Deformable attention intuition

### Implement
- [ ] Offset-based key sampling
- [ ] Attention over sampled keys only
- [ ] Sampling visualization

### Intuition
- Attend where it matters, not everywhere

### Modern Connection
- Deformable attention for detection & tracking

---

# Stage 8 — DETR-Style Detection

## Concept: Detection as set prediction

### Implement
- [ ] Image tokens from patches
- [ ] Learnable object queries
- [ ] Cross-attention (queries → image tokens)
- [ ] Linear heads for class & box
- [ ] Hungarian matching (NumPy)
- [ ] Toy IoU + L1 losses

### Intuition
- Object queries are competing explanations
- Matching resolves permutation ambiguity

### Modern Connection
- DETR, object queries, transformer decoders

---

# Stage 9 — Tracking & Video Memory

## Concept A: Tracking by attention

### Implement
- [ ] Cross-frame attention
- [ ] Association score matrix
- [ ] Greedy or Hungarian matching

### Intuition
- Tracking = correspondence over time

### Modern Connection
- Transformer-based tracking

---

## Concept B: Temporal memory

### Implement
- [ ] Memory token bank
- [ ] Current frame attends to memory
- [ ] Memory update strategies (FIFO, top-k)

### Intuition
- Long-term reasoning requires memory compression

### Modern Connection
- Video transformers with long-term memory

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
