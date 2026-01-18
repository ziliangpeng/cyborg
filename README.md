# Cyborg Lambda

A monorepo containing CUDA kernels, AI/LLM implementations, and visual processing applications.

[![Codecov](https://codecov.io/gh/ziliangpeng/cyborg/branch/main/graph/badge.svg)](https://codecov.io/gh/ziliangpeng/cyborg)

## Structure

- `ai/` - AI and LLM implementations
- `apps/` - Applications (halftone, cybervision, tokenizer_viz, etc.)
- `cuda/` - CUDA kernel implementations
- `visual/` - Visual processing utilities
- `experimental/` - Experimental code

## Building

This project uses [Bazel](https://bazel.build) as its build system.

```bash
# Run all tests
bazel test //...

# Build all targets
bazel build //...

# Run tests with coverage
bazel coverage --config=ci //...
```

## CI/CD

- Bazel tests run on every push to main and PRs
- Coverage is uploaded to Codecov automatically
- Playwright tests run for cybervision app

## License

MIT
