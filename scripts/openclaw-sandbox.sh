#!/bin/bash
# Run OpenClaw gateway as a background daemon in Docker
#
# Usage:
#   ./scripts/openclaw-sandbox.sh              # Build and start daemon
#   ./scripts/openclaw-sandbox.sh stop         # Stop the daemon
#   ./scripts/openclaw-sandbox.sh logs         # Tail logs
#   ./scripts/openclaw-sandbox.sh restart      # Restart the daemon
#   ./scripts/openclaw-sandbox.sh status       # Check if running

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker/openclaw-sandbox"

cd "$DOCKER_DIR"

case "${1:-start}" in
  stop)
    docker compose down
    ;;
  logs)
    shift
    docker compose logs -f "$@"
    ;;
  restart)
    docker compose down
    # Fetch latest OpenClaw version to bust Docker cache when it changes
    OPENCLAW_VERSION=$(npm view openclaw version 2>/dev/null || echo "latest")
    export OPENCLAW_VERSION
    docker compose build --build-arg OPENCLAW_VERSION="$OPENCLAW_VERSION"
    docker compose up -d
    echo "OpenClaw gateway restarted. Logs: ./scripts/openclaw-sandbox.sh logs"
    ;;
  start|"")
    # Fetch latest OpenClaw version to bust Docker cache when it changes
    OPENCLAW_VERSION=$(npm view openclaw version 2>/dev/null || echo "latest")
    export OPENCLAW_VERSION
    docker compose build --build-arg OPENCLAW_VERSION="$OPENCLAW_VERSION"
    docker compose up -d
    echo "OpenClaw gateway started on port 18789. Logs: ./scripts/openclaw-sandbox.sh logs"
    ;;
  status)
    docker compose ps
    ;;
  *)
    echo "Usage: $0 [start|stop|restart|logs|status]"
    exit 1
    ;;
esac
