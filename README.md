# Computer Vision Basics from Scratch

A **NumPy-only**, from-first-principles learning project to deeply understand
modern Computer Vision — from signal processing and compression
to transformers, masked modeling, detection, and tracking.

This repository is designed as a **conceptual curriculum**, not a production library.
The goal is to build *intuition*, not benchmarks.

---

## Philosophy

- Learn CV by **implementing the core ideas manually**
- Prefer **math, matrix operations, and data flow** over abstractions
- Connect:
  - signal processing
  - redundancy & compression
  - representation learning
  - attention & transformers
- Treat classical CV and modern deep learning as **one continuous system**

---

## What This Repo Is (and Is Not)

**This repo is:**
- A personal learning roadmap
- A reference for reading CV / ViT / MAE / DETR papers
- A place to experiment with ideas using small, interpretable examples

**This repo is NOT:**
- A deep learning framework
- A benchmark-driven implementation
- GPU-optimized or large-scale

---

## Constraints

- **NumPy only** for core computation
- ❌ No PyTorch / TensorFlow / JAX
- ❌ No autograd
- ❌ No GPU tricks
- Small images and toy videos are encouraged

---

## Repository Structure

Each stage corresponds to a conceptual milestone and is organized as:

```
NN_stage_name/
├── README.md    # concepts, equations, checklist, reflection
└── src/         # NumPy-only implementations
```

Shared reusable utilities live in:

```
utils/
└── src/
```

---

## Installation & Setup

This project uses **Conda** to manage the environment.

### 1. Create Conda Environment

```bash
conda create -n cv-basics python=3.10 -y
conda activate cv-basics
```

### 2. Install Core Dependencies

```bash
pip install -U pip
pip install numpy matplotlib pillow imageio
```

Optional (used in later stages):

```bash
pip install scipy
```

---

### 3. Development Tooling (Recommended)

This repo uses **pre-commit** to enforce formatting and linting.

```bash
pip install pre-commit black ruff
pre-commit install
```

Validate setup:

```bash
pre-commit run --all-files
```

---

### 4. Pre-commit Hooks

Configured hooks include:

- **Black** — code formatting
- **Ruff** — fast linting and import sorting
- Trailing whitespace & EOF checks
- Large file protection

All commits must pass these checks.

---

## Commit Convention

This repo follows **Conventional Commits v1.0.0**.

Format:

```
<type>(<scope>): <description>
```

Examples:

```
feat(05_patch_tokens_vit): add patchify and token projection
fix(attention): stabilize softmax computation
docs(ROADMAP): clarify MAE intuition
chore(repo): add pre-commit configuration
```

See `COMMIT_CONVENTION.md` for full details.

---

## How to Use This Repo

- Follow stages in numerical order
- Each stage README contains:
  - concept summary
  - implementation checklist
  - key equations
  - reflection on representation changes
- Focus on understanding **why** each component exists

---

## End Goal

After completing this project, you should be able to:

- Read ViT / MAE / DETR / tracking papers comfortably
- Reason about attention, sparsity, masking, and compression
- Explain modern CV models from first principles
- See classical CV and deep learning as one coherent system

---

> If you can implement it with NumPy, you truly understand it.
