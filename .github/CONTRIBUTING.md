# Contributing (Personal Workflow)

Even as a solo repo, follow this to keep the project consistent.

## Definition of Done (per stage)
A stage is considered "done" when it has:
- `README.md` including:
  - concept summary
  - checklist of implemented items
  - key equations (minimal but correct)
  - reflection: what changed in representation and why
- at least one runnable demo script (or notebook)
- at least one visualization that builds intuition
- shape annotations in key functions
- at least one sanity check (shape/numerical)

## Folder rules
- Stage-specific code goes in `NN_stage_name/src/`
- Shared code goes only in `utils/src/`
- Do not duplicate shared helpers across stages

## Python rules
- NumPy only for core math
- Keep code readable and explicit
- Prefer float32 for math-heavy ops

## Notes style
Write notes as if explaining to your future self before an exam:
- include shapes
- include a simple diagram in ASCII if helpful
- focus on "why this works"
