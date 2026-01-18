# Cyborg Project Memory

## Git Workflow

- **Always create a new branch from latest main** for any changes to this repo (fetch first to ensure latest: `git fetch origin main && git branch <new-branch> origin/main`)
- Never commit directly to main
- Use feature branches with descriptive names (e.g., `feat/halftone-image-processing`)

## Build System

- This monorepo uses [**Bazel**](https://bazel.build) for building and managing dependencies.
