# Commit Convention (Conventional Commits v1.0.0)

All commits must follow:
`<type>[optional scope][optional !]: <description>`

## Types
Use one of:
- feat:     new functionality (new module, algorithm, stage milestone)
- fix:      bug fix (math correction, shape bug, stability issue)
- docs:     documentation only
- refactor: restructure code without changing behavior
- test:     add/update sanity checks or tests
- chore:    repo maintenance (templates, gitignore, config)
- perf:     speed improvements without changing outputs
- style:    formatting only (no logic changes)

## Scopes (recommended, and treated as mandatory in this repo)
Use either:
- a stage folder: 00_numpy_basics, 01_image_representation, ..., 10_tracking_and_video_memory
- a subsystem: utils, attention, dct, convolution, patchify, mae, detr, tracking, video, viz, repo

## Description rules
- imperative mood: "add", "fix", "refactor", "document"
- short and clear (<= 72 chars preferred)
- describe the change, not the file name

## Breaking changes
If a change breaks existing usage, mark it with `!` or a footer:
- `refactor(utils)!: rename attention masking API`
or
- Footer: `BREAKING CHANGE: ...`

## Examples
- feat(02_convolution_signal_processing): implement conv2d loop baseline
- perf(convolution): vectorize conv2d using im2col
- fix(attention): stabilize softmax by subtracting max
- docs(03_frequency_dct_compression): add DCT energy compaction notes
- test(dct): add inverse reconstruction sanity check
- chore(repo): add copilot instructions and issue templates
