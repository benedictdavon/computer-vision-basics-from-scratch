# Stage 01 — Classical CV as Signal Processing

## Goal
Build intuition for:
- images as signals (representation matters)
- linear transforms on images
- convolution as structured feature extraction
- frequency-domain representations and energy compaction

This stage grounds computer vision in **signal processing**, forming the conceptual bridge between raw tensors (Stage 00) and learned vision systems.

---

## What You Will Create in This Stage

By the end of Stage 01, you should have:

### 1) A signal-based image representation module
A set of NumPy-only utilities and demos for:
- image tensor formats across space, batch, and time
- color-space and representation transforms as linear maps
- pixel statistics and distribution diagnostics
- resizing, interpolation, and aliasing intuition

---

### 2) A convolution & filtering module (from scratch)
A fully transparent implementation of classical convolution:
- naive loop-based convolution
- multi-channel and multi-filter convolution
- vectorized convolution via im2col + GEMM
- classical hand-designed kernels
- edge magnitude and orientation computation
- separable convolution and efficiency intuition

---

### 3) A frequency-domain analysis module
A minimal but rigorous exploration of frequency representations:
- DCT as a separable linear transform
- energy compaction in natural images
- top-k reconstruction and compression intuition
- comparison with FFT for interpretation

---

### 4) One stage `main.py` mini project
A runnable script that:
- loads a small set of real images
- applies representation → spatial → frequency transforms
- prints shape and statistic traces
- produces visual diagnostics at each step

---

## Concepts Covered

- Images as discrete signals
- Linear transforms on pixel spaces
- Representation contracts (shape, dtype, statistics)
- Sampling, interpolation, and aliasing
- Convolution as local dot-product scanning
- Inductive bias: locality & translation equivariance
- Classical edge detection and gradients
- Frequency-domain representations (DCT, FFT)
- Energy compaction and reconstruction priors

---

## Implementation Checklist

### A) Image Representation & Signal View
- [ ] Handle image tensor formats:
  - `(H, W)` grayscale
  - `(H, W, C)` color
  - `(B, H, W, C)` batch
  - `(T, H, W, C)` temporal signals
- [ ] Enforce a consistent internal layout (default: HWC)
- [ ] Enforce dtype/range contracts (`uint8` vs `float32`)
- [ ] Implement RGB → grayscale (luma-style)
- [ ] Implement a simple YCbCr-like color transform using matrix multiplication
- [ ] (Optional) Toy chroma subsampling (4:2:0-style downsample/upsample)
- [ ] Compute pixel-wise statistics (min/max/mean/std)
- [ ] Plot per-channel histograms
- [ ] Implement nearest-neighbor resize
- [ ] Implement bilinear resize
- [ ] Demonstrate aliasing via downsampling

---

### B) Convolution as Feature Extraction
- [ ] Implement naive 2D convolution (loops, grayscale)
- [ ] Add explicit padding logic (`valid` / `same`)
- [ ] Extend convolution to multi-channel inputs (RGB)
- [ ] Support multiple output filters
- [ ] Implement vectorized convolution using im2col
- [ ] Verify numerical equivalence between naive and vectorized versions
- [ ] Implement classical kernels:
  - blur / box filter
  - Sobel-x
  - Sobel-y
  - Laplacian
- [ ] Compute edge magnitude from gradient responses
- [ ] Compute edge orientation using `arctan2`
- [ ] Implement separable convolution
- [ ] Demonstrate computational efficiency intuition (small timing demo)

---

### C) Frequency Domain & Energy Compaction
- [ ] Implement 1D DCT-II from definition
- [ ] Implement 2D DCT via separability
- [ ] Visualize DCT coefficient energy maps
- [ ] Sort coefficients by magnitude
- [ ] Reconstruct images using top-k coefficients
- [ ] Compare reconstruction quality vs k
- [ ] Compare DCT vs FFT representations (interpretation-focused)
- [ ] Explain why DCT concentrates energy better for natural images

---

### D) Visualization & Diagnostics
- [ ] Print shape / dtype / range after every major transform
- [ ] Plot original vs transformed images
- [ ] Plot histograms before vs after transforms
- [ ] Visualize kernel responses side-by-side
- [ ] Visualize frequency-domain coefficient heatmaps
- [ ] Assert invariants (no NaNs, valid ranges, expected shapes)

---

## Mini Project — “Signals → Features → Frequencies”

**Purpose:**
Repeatedly practice viewing images as signals under different linear transforms,
and observe what information is preserved or discarded.

This mini project prepares intuition for:
- CNN inductive bias
- learned filters vs hand-crafted filters
- patch embeddings and reconstruction
- frequency-aware reasoning in modern models

---

## Mini Project Outputs (What `main.py` should do)

Your `main.py` should:

### 1) Load a small set of real images
- Load images as:
  - `images_uint8: (B, H, W, C)`
- Convert to:
  - `images_f32: float32 in [0, 1]`

---

### 2) Print “signal traces”
After each major step, print:
- tensor name
- shape
- dtype
- min / max / mean / std

---

### 3) Representation demos
- RGB → grayscale
- RGB → YCbCr-like transform
- histogram comparison across channels
- resize + downsample aliasing demo

---

### 4) Convolution demos
- apply blur / Sobel / Laplacian kernels
- visualize filter responses
- compute and visualize edge magnitude + orientation
- compare naive vs im2col convolution outputs

---

### 5) Frequency demos
- compute 2D DCT of grayscale image
- visualize coefficient energy
- reconstruct with top-k coefficients
- compare original vs reconstructed images

---

### 6) Validate correctness
- [ ] Assert shape invariants at each stage
- [ ] Verify naive vs vectorized convolution equivalence
- [ ] Verify reconstruction error decreases as k increases

---

## Suggested Files to Create (Stage 01)

01_signal_processing/
├── src/
│ ├── a_representation/
│ │ ├── formats.py
│ │ ├── color.py
│ │ ├── stats.py
│ │ └── resize.py
│ ├── b_convolution/
│ │ ├── padding.py
│ │ ├── conv_naive.py
│ │ ├── conv_multi.py
│ │ ├── im2col.py
│ │ ├── conv_im2col.py
│ │ ├── kernels.py
│ │ └── edges.py
│ └── c_frequency/
│   ├── dct1d.py
│   ├── dct2d.py
│   ├── energy.py
│   └── recon.py
├── demos/
├── assets/
└── main.py

---

## Key Notes
- Representation choices are not neutral.
- Convolution is linear algebra with structure and bias.
- Frequency explains compression, denoising, and reconstruction.
- If you can’t explain a filter response, you don’t understand it yet.

---

## Reflection (Write this after completing Stage 01)
Answer in your own words:
- How did representation changes affect downstream results?
- What information is lost during sampling or resizing?
- Why does convolution encode a useful inductive bias?
- Why do natural images compress so well in the DCT domain?
- Which intuitions here connect most clearly to CNNs or ViTs?
