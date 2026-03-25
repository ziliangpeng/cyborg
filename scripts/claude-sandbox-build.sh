#!/bin/bash
# Build the Claude Code sandbox Docker image
#
# Usage:
#   ./scripts/claude-sandbox-build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker/claude-sandbox"

# Fetch latest Claude Code version
CLAUDE_CODE_VERSION=$(npm view @anthropic-ai/claude-code version 2>/dev/null || echo "latest")
export CLAUDE_CODE_VERSION

echo "Building claude-sandbox with Claude Code v${CLAUDE_CODE_VERSION}..."
cd "$DOCKER_DIR"
docker compose build --build-arg CLAUDE_CODE_VERSION="$CLAUDE_CODE_VERSION"
echo "Done."
