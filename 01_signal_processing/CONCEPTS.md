# Stage 01 — CONCEPTS.md
#
# Purpose:
# - Treat images as signals and classical CV as (mostly) linear transforms.
# - Make representation choices explicit: layout, dtype, range, sampling, color space.
# - Build “signal intuition” that transfers directly to CNNs, ViTs, MAE, DETR.
# - Keep this file framework-agnostic (NumPy-first, but concepts generalize).

---

## 0) How to Use This File

Stage 01 code is your "signal lab."
This `CONCEPTS.md` is your "signal compass."

When something looks “off” later (edges wrong, colors weird, filters unstable, reconstructions muddy), revisit:
- **Representation contracts** (shape/layout/dtype/range)
- **Sampling & interpolation** (resize, downsample, aliasing)
- **Convolution definitions** (correlation vs convolution, padding/stride)
- **Derivatives** (Sobel/Laplacian amplify noise)
- **Frequency intuition** (energy compaction, reconstruction)

If you can reason correctly about those five, most classical CV becomes “just linear algebra on signals.”

---

## 1) Images as Signals (and Why Representation Matters More Than You Think)

### What is an image (in this project)?
An image is a discrete signal sampled on a 2D grid.
It is a tensor with:
- a **shape** (axes + sizes),
- a **dtype** (numeric representation),
- a **range/statistics** (the “data contract”),
- a **layout convention** (where channels/time/batch live).

### Common CV shapes you must recognize instantly

**Single images**
- Grayscale: `(H, W)`
- RGB (channel-last): `(H, W, C)` with `C=3`
- RGBA: `(H, W, 4)`

**Batches**
- Batch of RGB images: `(B, H, W, C)` (BHWC)

**Time / video**
- Video: `(T, H, W, C)`
- Batched video (if needed): `(B, T, H, W, C)`

### Why representation matters
A model or a pipeline never sees “the picture.”
It sees **numbers** under a contract:
- channel order (RGB vs BGR),
- scaling ([0,255] vs [0,1]),
- normalization (mean/std),
- resize method (nearest vs bilinear vs filtered),
- color transform (RGB vs grayscale vs YCbCr-like).

Rule:
> If you can’t state the contract, you can’t debug the pipeline.

---

## 2) Axes: (y, x) Is Not (x, y)

### Pixel indexing convention (non-negotiable)
For an array `img`:
- `img[y, x]` is the pixel at row `y` and column `x`

Where:
- `y` increases downward (top → bottom)
- `x` increases rightward (left → right)

### Why this matters immediately
- Convolution loops are `(y, x)` loops.
- Sobel-x means “change along x” (columns).
- Crops are `img[y1:y2, x1:x2]`, not `img[x1:x2, y1:y2]`.

Rule:
> Treat axis-meaning as semantic, not cosmetic.

---

## 3) Dtypes & Ranges: uint8 Is a Storage Format, Not a Math Format

### Why images are uint8
Most image files store pixels as `uint8`:
- compact
- fast I/O
- values in `[0, 255]`

### Why uint8 breaks math
uint8 arithmetic can overflow/wrap.
So pipelines do:
1) convert to float32
2) scale to `[0,1]`
3) operate in float space

Canonical conversions:
- `uint8 -> float32`: `img_f = img_u8.astype(np.float32) / 255.0`
- `float32 -> uint8`: `img_u8 = clip(round(img_f * 255), 0, 255).astype(uint8)`

Rule:
> Never do meaningful filtering/gradients/convolution in uint8.

### Statistics are part of the contract
Always track:
- min/max
- mean/std (per-channel)
- histogram shape (exposure/contrast)

If these drift unexpectedly, your pipeline will “work” but be wrong.

---

## 4) Linear Transforms on Pixels: Color & Representation as Matrix Multiplication

Many “representation changes” are **linear maps** applied per pixel.

### Per-pixel linear map (color transform)
Let RGB pixel be a vector `p = [R, G, B]^T`.
A linear transform is:
- `q = M p` where `M` is 3×3.

This is the same conceptual tool as:
- 1×1 convolutions (per-pixel channel mixing)
- learned “color mixing” layers early in models

Rule:
> A surprising amount of CV is “apply a linear map everywhere.”

---

## 5) RGB ↔ Grayscale: Luma-Style Conversion

### Mental model
Grayscale is a weighted sum because human perception is not uniform across channels:
- green contributes more than blue

A common luma-like approximation:
- `Y = 0.299 R + 0.587 G + 0.114 B`

### Common bugs
- using wrong channel order (BGR treated as RGB)
- doing the weighted sum in uint8 (precision/overflow issues)
- forgetting grayscale shape becomes `(H, W)` (not `(H, W, 1)` unless you choose so)

Sanity checks:
- If `R=G=B` everywhere, grayscale equals that value (after scaling).
- For a purely red region, grayscale should be ~0.299 of red (in normalized space).

---

## 6) YCbCr-like Transform (Toy, but Deeply Useful)

### Mental model
Separate:
- **luma** (brightness structure → edges live here)
- **chroma** (color differences → less spatial detail needed)

Toy approach:
- compute `Y` as luma
- compute two chroma channels as differences from `Y`

Key lesson:
> Many pipelines preserve “structure” mostly in luma.

Modern connection:
- preprocessing assumptions in ViT/MAE/DETR are effectively “representation priors.”

---

## 7) (Optional) Chroma Subsampling (Toy 4:2:0 Intuition)

### Mental model
Chroma is stored at lower resolution than luma.
Toy 4:2:0:
- keep `Y` at `(H, W)`
- downsample `Cb` and `Cr` to `(H/2, W/2)`
- upsample back when reconstructing

What you learn:
- color detail degrades before edge detail
- downsample/upsample introduces blur/artifacts in chroma

Sanity check:
- edges remain mostly intact, but fine color boundaries smear.

---

## 8) Pixel Statistics & Histograms: Diagnostics, Not Decoration

### Why histograms matter
Histograms tell you:
- exposure (too dark/bright)
- contrast (narrow vs wide spread)
- clipping (spikes at 0 or 255)
- normalization correctness (after mean/std, distribution recenters)

### Common pitfalls
- comparing histograms across different ranges ([0,255] vs [0,1]) without matching bins
- mixing channel histograms without labeling

Sanity checks:
- histogram counts sum to number of pixels (per channel)
- mean/std from histogram roughly match direct computation

Rule:
> If the histogram surprises you, your pipeline is lying somewhere.

---

## 9) Resizing & Interpolation: Resampling a Signal (Not “Stretching Pixels”)

Resizing changes sampling density. That changes what information survives.

### Nearest neighbor
- picks nearest source pixel
- blocky
- aliasing-prone
- fast

Mental model:
> “copy or drop pixels”

### Bilinear
- weighted average of 4 neighbors
- smoother
- still not a true low-pass filter

Mental model:
> “local plane approximation”

### Aliasing (the key concept)
Downsampling without anti-alias filtering folds high frequencies into low:
- moiré patterns
- false edges
- weird textures

Rule:
> Proper downsampling is usually: low-pass filter → then subsample.

Sanity checks:
- constant image stays constant after resize
- resize down then up: bilinear blurs; nearest blocks

---

## 10) Convolution: Local Dot-Product Scanning (Structured Linear Layer)

### Core mental model
Convolution is:
> sliding a kernel across the image and computing local dot products.

For grayscale `I` and kernel `K` (correlation form):
- `O[y, x] = Σ_i Σ_j K[i, j] * I[y+i, x+j]`

This is a **linear operator** over the image signal.

### Convolution vs correlation (don’t mix them!)
- **Convolution** flips the kernel
- **Correlation** does not

Most deep learning layers implement correlation but call it convolution.
For Stage 01:
- choose one definition
- document it once
- keep it consistent across naive + im2col + separable

Rule:
> Mismatch here causes “same code, different results” bugs.

---

## 11) Padding, Stride, and Output Shapes (the Plumbing That Breaks Everything)

### Padding modes (conceptually)
- `valid`: no padding → output shrinks
- `same`: pad so output spatial size matches input (stride=1 typical)

### Output size formula (per dimension)
Given:
- input size `N`
- kernel size `F`
- padding `P`
- stride `S`

Output:
- `out = floor((N + 2P - F)/S) + 1`

Apply to H and W.

Sanity checks:
- `same` + stride 1 → output shape equals input shape
- `valid` + stride 1 → output shrinks by `F-1`

Common bugs:
- padding wrong axis (mix H/W)
- off-by-one errors in loops
- padding differs between naive and im2col implementation

Rule:
> Test on tiny arrays where you can compute output by hand.

---

## 12) Multi-Channel Convolution (RGB In, Many Filters Out)

### Mental model
A filter spans all input channels.
At each location, output is:
- sum of (kernel * patch) across spatial dims and across channels

If input is `(H, W, C)` and filter is `(kH, kW, C)`, output is `(H_out, W_out)`.
With `F` filters, output is `(H_out, W_out, F)`.

Modern connection:
- This is literally a CNN layer:
  - input channels → output channels via multiple kernels

Common bugs:
- forgetting to sum over channels
- wrong broadcasting when multiplying patch and kernel
- mixing layouts (HWC vs CHW)

Sanity checks:
- if kernel is zero everywhere except one channel, output should depend only on that channel
- compare grayscale conv vs RGB conv on grayscale-replicated RGB (should match if kernels identical per channel)

---

## 13) Classical Kernels: What They Detect (Signal Interpretation)

### Blur / box filter (low-pass)
- averages neighborhood
- removes high frequencies (noise, fine texture)
- reduces sharp edges

Mental model:
> “smooth the signal”

### Sobel (first derivatives / gradients)
- `Sobel_x`: responds to changes along x (vertical edges)
- `Sobel_y`: responds to changes along y (horizontal edges)

Mental model:
> “local slope of intensity”

### Laplacian (second derivative)
- highlights rapid change regions
- used for sharpening / edge emphasis

Mental model:
> “local curvature”

Important note:
> Derivative filters amplify noise. Blur before gradients is common.

---

## 14) Edge Magnitude & Orientation: From Filters to Geometry

Given gradient responses:
- `Gx` from Sobel-x
- `Gy` from Sobel-y

Compute:
- magnitude: `M = sqrt(Gx^2 + Gy^2)`
- orientation: `θ = arctan2(Gy, Gx)`

Key points:
- `arctan2` handles quadrant correctly (don’t use `arctan(Gy/Gx)`).
- magnitude is “edge strength”
- orientation is “direction of greatest increase” (gradient direction)

Sanity checks:
- constant image → near-zero gradients everywhere
- vertical step edge → strong `Gx`, weak `Gy`

---

## 15) Separable Convolution: Structure + Efficiency

A kernel `K` is separable if:
- `K = a ⊗ b` (outer product of two 1D kernels)

Then you can do:
1) convolve with `a` in one direction
2) convolve with `b` in the other

Cost intuition:
- full 2D: O(k^2) per pixel
- separable: O(2k) per pixel

Modern connection:
- factorization ideas appear in efficient CNNs
- “do something structured instead of dense” is the same spirit as attention factorization tricks

Sanity checks:
- separable implementation should match full 2D (within tolerance)
- test on random small images + known separable kernels

---

## 16) im2col + GEMM: Convolution as Matrix Multiplication

### Mental model
Convolution becomes GEMM if you:
- extract every local patch and flatten it → a big matrix `X_col`
- flatten kernels → matrix `W`
- multiply: `Y = X_col @ W`

This reveals:
- convolution is a structured linear layer
- weight sharing + locality are baked into how patches are formed

Tradeoffs:
- fast if GEMM is optimized
- memory heavy (patch matrix can be huge)

Sanity checks:
- naive vs im2col outputs match for multiple random tests
- padding behavior identical
- verify shapes at each step:
  - `X_col: (H_out*W_out, kH*kW*C)`
  - `W: (kH*kW*C, F)`
  - `Y: (H_out*W_out, F)` reshape to `(H_out, W_out, F)`

Rule:
> If shapes don’t make sense, the implementation cannot be correct.

---

## 17) Frequency Domain & Energy Compaction (Why DCT Exists in Vision)

### Why frequency at all?
Spatial domain tells you “where values are.”
Frequency tells you “how fast values change.”

- Low frequency: smooth illumination, large shapes
- High frequency: edges, fine texture, noise

Natural images usually concentrate energy in low frequencies.

---

## 18) DCT: Images as Sums of Cosine Basis Patterns

### Mental model
DCT represents a signal as:
> “how much of each cosine pattern is present”

DCT is popular for images because:
- real-valued basis
- strong energy compaction for natural images
- good boundary behavior for block-based coding

### 1D DCT-II (conceptual definition)
For `x[n]` length N:
- `X[k] = Σ x[n] * cos( π/N * (n + 1/2) * k )` (ignoring normalization for intuition)

Key intuition:
- `k=0` is the DC term (average level)
- larger k = faster oscillations

---

## 19) 2D DCT via Separability (Same “Factorization Mindset”)

2D DCT can be computed as:
1) DCT along rows
2) DCT along columns

This is separability again:
- strong conceptual tie to separable convolution

Sanity checks:
- constant image → only DC coefficient significant
- smooth gradient → energy clustered in low-frequency corner

Visualization tip:
- use `log(1 + abs(coeff))` to see structure (DC dominates otherwise)

---

## 20) Top-k Reconstruction: Compression Intuition You Can Feel

### Mental model
If energy is concentrated, you can keep only large coefficients:
- zero out the rest
- invert DCT
- get an approximation

Key properties:
- k=1 (only DC) → flat image
- increasing k should monotonically reduce reconstruction error (for consistent selection rule)

Optional metrics:
- MSE
- PSNR (interpretability)

Rule:
> Reconstruction quality is a direct readout of “how compact” the signal is.

---

## 21) DCT vs FFT (Interpretation, Not a Competition)

### FFT/DFT
- complex coefficients (magnitude + phase)
- sinusoidal basis with circular boundary assumptions in simplest form

### DCT
- real cosines
- tends to compact energy well for typical image blocks
- widely used in compression (JPEG-style intuition)

Takeaway:
> DCT is a practical frequency basis for images; FFT is a more general spectral tool.

---

## 22) Connecting Stage 01 to Modern Vision

### CNNs
- learn filters instead of hand-designed kernels
- early layers often resemble edge/texture detectors
- im2col view shows: conv layer = structured linear map + nonlinearity

### ViT / patch embeddings
- patchify + linear projection creates token representations
- representation choices (resize, normalization) strongly affect token statistics
- early learned projections often encode low-frequency structure + edge-like patterns

### MAE
- reconstruction ties to priors: natural images are structured and low-frequency heavy
- DCT energy compaction explains why “a small set of components explains a lot”

### Deformable attention
- adaptive sampling resembles learned receptive fields
- shares the spirit of “local processing,” but with dynamic offsets and content-dependent sampling

---

## 23) Debugging Rules (Non-Negotiable)

1) **Make the contract visible**
   - print shape, dtype, min/max/mean/std after major steps

2) **Test tiny cases**
   - 5×5 image, 3×3 kernel
   - compute one output pixel by hand

3) **Use impulse tests**
   - impulse image convolved with kernel should “reproduce” kernel footprint

4) **Verify equivalence**
   - naive conv vs im2col conv within tolerance
   - full 2D conv vs separable conv within tolerance (for separable kernels)

5) **Control scale**
   - gradients and Laplacian can blow up magnitude; clip only for visualization, not for correctness

6) **Be explicit about definitions**
   - convolution vs correlation, padding style, output indexing

---

## 24) Common Bugs & Failure Modes (Stage 01)

- **Axis swap**: using (x,y) when code expects (y,x)
- **Layout mismatch**: HWC treated as CHW or vice versa
- **uint8 math**: overflow, wrap-around, nonsense gradients
- **Padding mismatch**: naive and im2col disagree at borders
- **Kernel orientation confusion**: convolution vs correlation flip
- **Arctan instability**: using arctan instead of arctan2
- **Aliasing surprise**: downsampling produces patterns you didn’t “create”
- **Frequency plots look blank**: DC dominates; must log-scale

---

## 25) Checklist: “Do I Really Understand Stage 01?”

You understand Stage 01 if you can explain, without guessing:

**Representation**
- what your default image contract is (shape/layout/dtype/range)
- why resizing is resampling (and why aliasing happens)
- why luma/chroma separation is meaningful

**Convolution**
- the exact definition you implemented (correlation vs convolution)
- how padding changes output shape
- how multi-channel conv sums across channels
- why Sobel approximates derivatives and why it amplifies noise
- why separable conv is faster and when it matches full 2D

**Frequency**
- what the DC term represents
- why natural images compact energy in low frequencies
- why top-k reconstruction improves with k
- what DCT vs FFT means conceptually (real cosine basis vs complex spectral basis)

If any of these feel fuzzy, revisit the demos and re-trace shapes + contracts.
