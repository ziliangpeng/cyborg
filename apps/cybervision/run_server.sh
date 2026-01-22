#!/bin/bash
echo "Building and starting CyberVision server..."
bazel run //apps/cybervision:cybervision -- --port 8080
