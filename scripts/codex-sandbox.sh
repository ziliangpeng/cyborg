#!/bin/bash
# Run OpenAI Codex CLI in a sandboxed Docker container
#
# Usage:
#   ./scripts/codex-sandbox.sh              # Interactive Codex session
#   ./scripts/codex-sandbox.sh "prompt"     # Non-interactive mode
#
# First time setup:
#   1. Run this script once
#   2. Select "Sign in with ChatGPT" when prompted
#   3. Credentials will be stored in ~/.codex/sandboxed-home/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker/codex-sandbox"

# Build and run
cd "$DOCKER_DIR"
docker compose build
docker compose run --rm codex "$@"
