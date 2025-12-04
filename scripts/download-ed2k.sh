#!/bin/bash

# Script to download ed2k links using aMule in Docker
# Usage: ./download-ed2k.sh <command> [args]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_DIR/data"
CONFIG_DIR="$DATA_DIR/amule-config"
DOWNLOADS_DIR="$DATA_DIR/downloads"
CONTAINER_NAME="cyborg-amule"

# Create necessary directories
mkdir -p "$CONFIG_DIR" "$DOWNLOADS_DIR"

# Get password from container logs
get_password() {
    docker logs "$CONTAINER_NAME" 2>&1 | grep "No GUI password specified" | head -1 | sed 's/.*using generated one: //' | xargs
}

# Ensure container is running
ensure_container_running() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container not running. Starting..."

        if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            docker start "$CONTAINER_NAME"
            sleep 5
        else
            echo "Creating new container..."
            curl -s -o "$CONFIG_DIR/server.met" "http://shortypower.org/server.met" 2>/dev/null || echo "Warning: Could not download fresh server list"

            docker run -d \
                --name "$CONTAINER_NAME" \
                --network host \
                -v "$CONFIG_DIR:/home/amule/.aMule" \
                -v "$DOWNLOADS_DIR:/incoming" \
                tchabaud/amule

            echo "Waiting for aMule to initialize..."
            sleep 10
        fi
    fi
}

# Command: add
cmd_add() {
    if [ -z "$1" ]; then
        echo "Usage: $0 add <ed2k-link>"
        exit 1
    fi

    ED2K_LINK="$1"
    echo "=== Adding ed2k Download ==="
    echo "Link: $ED2K_LINK"
    echo ""

    ensure_container_running

    PASSWORD=$(get_password)
    if [ -z "$PASSWORD" ]; then
        echo "Error: Could not extract password from container logs"
        exit 1
    fi

    echo "Adding to download queue..."
    docker exec "$CONTAINER_NAME" amulecmd -h localhost -p 4712 -P "$PASSWORD" -c "Add $ED2K_LINK"

    echo ""
    echo "Current downloads:"
    docker exec "$CONTAINER_NAME" amulecmd -h localhost -p 4712 -P "$PASSWORD" -c "Show DL"
}

# Command: status
cmd_status() {
    echo "=== Download Status ==="
    echo ""

    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is not running"
        exit 0
    fi

    PASSWORD=$(get_password)
    if [ -z "$PASSWORD" ]; then
        echo "Error: Could not extract password"
        exit 1
    fi

    docker exec "$CONTAINER_NAME" amulecmd -h localhost -p 4712 -P "$PASSWORD" -c "Show DL"
}

# Command: log
cmd_log() {
    LINES="${1:-50}"
    echo "=== aMule Container Logs (last $LINES lines) ==="
    echo ""

    if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container does not exist"
        exit 1
    fi

    docker logs "$CONTAINER_NAME" 2>&1 | tail -"$LINES"
}

# Command: start
cmd_start() {
    echo "=== Starting Container ==="

    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is already running"
        exit 0
    fi

    ensure_container_running
    echo "Container started successfully"
}

# Command: stop
cmd_stop() {
    echo "=== Stopping Container ==="

    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is not running"
        exit 0
    fi

    docker stop "$CONTAINER_NAME"
    echo "Container stopped"
}

# Command: stats
cmd_stats() {
    echo "=== Download Statistics ==="
    echo ""

    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is not running"
        exit 0
    fi

    PASSWORD=$(get_password)
    if [ -z "$PASSWORD" ]; then
        echo "Error: Could not extract password"
        exit 1
    fi

    docker exec "$CONTAINER_NAME" amulecmd -h localhost -p 4712 -P "$PASSWORD" -c "Statistics"
}

# Show help
show_help() {
    cat << EOF
aMule ed2k Downloader - Docker wrapper for easy ed2k downloads

Usage: $0 <command> [args]

Commands:
  add <ed2k-link>     Add a new ed2k link to download queue
  status              Show current downloads and progress
  log [lines]         Show container logs (default: 50 lines)
  stats               Show download statistics
  start               Start the aMule container
  stop                Stop the aMule container
  help                Show this help message

Examples:
  $0 add "ed2k://|file|example.rar|1000000|HASH|/"
  $0 status
  $0 log 100
  $0 stats
  $0 start
  $0 stop

Downloads location: $DOWNLOADS_DIR
Config location: $CONFIG_DIR
EOF
}

# Main
COMMAND="${1:-help}"

case "$COMMAND" in
    add)
        cmd_add "$2"
        ;;
    status)
        cmd_status
        ;;
    log)
        cmd_log "$2"
        ;;
    stats)
        cmd_stats
        ;;
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        show_help
        exit 1
        ;;
esac
