#!/bin/bash
# Run Claude Code in a sandboxed Docker container
#
# Usage:
#   ./scripts/claude-sandbox.sh              # Interactive Claude session
#   ./scripts/claude-sandbox.sh --model opus # Pass extra args to Claude
#   ./scripts/claude-sandbox.sh -p "query"   # Non-interactive mode
#
# First time setup:
#   1. Run 'claude setup-token' on your host machine
#   2. Export the token: export CLAUDE_SETUP_TOKEN=<your-token>
#   3. Or add to your shell rc file for persistence

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker/claude-sandbox"

# Fetch latest Claude Code version to bust Docker cache when it changes
CLAUDE_CODE_VERSION=$(npm view @anthropic-ai/claude-code version 2>/dev/null || echo "latest")
export CLAUDE_CODE_VERSION

# Build and run
export HOST_CWD="$(pwd)"
cd "$DOCKER_DIR"
docker compose build --build-arg CLAUDE_CODE_VERSION="$CLAUDE_CODE_VERSION"
docker compose run --rm claude "$@"
