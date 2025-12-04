#!/bin/bash
set -e

# Script to download ed2k links using aMule in Docker
# Usage: ./download-ed2k.sh "ed2k://|file|filename|size|hash|/"

if [ -z "$1" ]; then
    echo "Usage: $0 <ed2k-link>"
    echo "Example: $0 'ed2k://|file|example.zip|12345|ABC123|/'"
    exit 1
fi

ED2K_LINK="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_DIR/data"
CONFIG_DIR="$DATA_DIR/amule-config"
DOWNLOADS_DIR="$DATA_DIR/downloads"
CONTAINER_NAME="cyborg-amule"

# Create necessary directories
mkdir -p "$CONFIG_DIR" "$DOWNLOADS_DIR"

echo "=== aMule ed2k Downloader ==="
echo "Data directory: $DATA_DIR"
echo "Downloads will be saved to: $DOWNLOADS_DIR"
echo ""

# Download fresh server list before container starts
echo "Downloading fresh server list..."
curl -s -o "$CONFIG_DIR/server.met" "http://shortypower.org/server.met" 2>/dev/null || echo "Warning: Could not download fresh server list"

echo ""

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' already exists."

    # Check if it's running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is already running."
    else
        echo "Starting existing container..."
        docker start "$CONTAINER_NAME"
        sleep 5
    fi
else
    echo "Creating and starting new aMule container..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --network host \
        -v "$CONFIG_DIR:/home/amule/.aMule" \
        -v "$DOWNLOADS_DIR:/incoming" \
        tchabaud/amule

    echo "Waiting for aMule to initialize..."
    sleep 10
fi

echo ""
echo "Extracting aMule password from logs..."
PASSWORD=$(docker logs "$CONTAINER_NAME" 2>&1 | grep "No GUI password specified" | head -1 | sed 's/.*using generated one: //' | xargs)

if [ -z "$PASSWORD" ]; then
    echo "Error: Could not extract password from container logs"
    exit 1
fi

echo "Waiting for aMule to process new server list..."
sleep 5

echo "Adding ed2k link to download queue..."
docker exec "$CONTAINER_NAME" amulecmd -h localhost -p 4712 -P "$PASSWORD" -c "Add $ED2K_LINK"

echo ""
echo "Current downloads:"
docker exec "$CONTAINER_NAME" amulecmd -h localhost -p 4712 -P "$PASSWORD" -c "Show DL"

echo ""
echo "=== Download started! ==="
echo ""
echo "To monitor progress, first extract the password:"
echo "  PASSWORD=\$(docker logs $CONTAINER_NAME 2>&1 | grep 'No GUI password specified' | sed 's/.*using generated one: //')"
echo "  docker exec $CONTAINER_NAME amulecmd -h localhost -p 4712 -P \"\$PASSWORD\" -c 'Show DL'"
echo ""
echo "To stop the container:"
echo "  docker stop $CONTAINER_NAME"
echo ""
echo "To remove the container:"
echo "  docker rm $CONTAINER_NAME"
echo ""
echo "Downloads location: $DOWNLOADS_DIR"
