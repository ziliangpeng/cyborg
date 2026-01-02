workspace(name = "cyborg")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# ============================================================================
# Platform Rules (required by rules_cuda)
# ============================================================================

http_archive(
    name = "platforms",
    urls = ["https://github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz"],
    sha256 = "218efe8ee736d26a3572663b374a253c012b716d8af0c07e842e82f238a0a7ee",
)

# ============================================================================
# CUDA Rules (community standard)
# ============================================================================
# https://github.com/bazel-contrib/rules_cuda

http_archive(
    name = "rules_cuda",
    sha256 = "fe8d3d8ed52b9b433f89021b03e3c428a82e10ed90c72808cc4988d1f4b9d1b3",
    strip_prefix = "rules_cuda-v0.2.5",
    urls = ["https://github.com/bazel-contrib/rules_cuda/releases/download/v0.2.5/rules_cuda-v0.2.5.tar.gz"],
)

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")

rules_cuda_dependencies()

# Auto-detect CUDA installation on the system
# This will find CUDA in standard locations (/usr/local/cuda, etc.)
register_detected_cuda_toolchains()
