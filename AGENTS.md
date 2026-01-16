# Cyborg Project Memory

## Git Workflow

- **Always create a new branch from latest main** for any changes to this repo (but avoid switching to main locally - use `git branch <new-branch> origin/main` or similar)
- Never commit directly to main
- Use feature branches with descriptive names (e.g., `feat/halftone-image-processing`)

## Build System

- This monorepo is officially managed by **Bazel** for build and dependency management
