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

# Build and run
cd "$DOCKER_DIR"
docker compose build --quiet
docker compose run --rm claude "$@"
