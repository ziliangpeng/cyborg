#!/bin/sh

# This script runs all runnable binaries within this package.

# Find the directory where this script is located
SCRIPT_DIR=$(dirname "$0")

for bin_file in "$SCRIPT_DIR"/src/bin/*.rs; do
    bin_name=$(basename "$bin_file" .rs)
    echo "--- Running $bin_name ---"
    ~/.cargo/bin/cargo run --bin "$bin_name" -p project-euler
    echo "-------------------------"
done
