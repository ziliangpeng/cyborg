#!/bin/bash
# Wrapper script for softmax_triton_demo that sets up CUDA library paths
#
# Usage:
#   bazel run //cuda/softmax:run_triton_demo -- -n 1024
#
# This script finds all NVIDIA CUDA libraries in the Bazel runfiles and
# adds them to LD_LIBRARY_PATH before running the Python demo.

set -e

# Find our runfiles directory (run_triton_demo.runfiles)
# Bazel sets RUNFILES_DIR or we can derive it from $0
if [[ -n "${RUNFILES_DIR:-}" ]]; then
    OUR_RUNFILES="$RUNFILES_DIR"
elif [[ -d "${0}.runfiles" ]]; then
    OUR_RUNFILES="${0}.runfiles"
else
    # Fallback: derive from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    OUR_RUNFILES="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
fi

# The softmax_triton_demo binary and its runfiles are siblings to our runfiles
BINARY_DIR="$(dirname "$OUR_RUNFILES")"
DEMO_BINARY="$BINARY_DIR/softmax_triton_demo"
DEMO_RUNFILES="$BINARY_DIR/softmax_triton_demo.runfiles"

if [[ ! -x "$DEMO_BINARY" ]]; then
    echo "Error: Could not find softmax_triton_demo binary at $DEMO_BINARY" >&2
    exit 1
fi

# Find all lib directories in the demo's runfiles (including nvidia packages)
LIB_PATHS=""
while IFS= read -r -d '' lib_dir; do
    if [[ -z "$LIB_PATHS" ]]; then
        LIB_PATHS="$lib_dir"
    else
        LIB_PATHS="$LIB_PATHS:$lib_dir"
    fi
done < <(find "$DEMO_RUNFILES" -path "*/site-packages/*/lib" -type d -print0 2>/dev/null)

# Export LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$LIB_PATHS:${LD_LIBRARY_PATH:-}"

# Run the demo
exec "$DEMO_BINARY" "$@"
