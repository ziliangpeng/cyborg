#!/bin/bash
# Download MediaPipe Selfie Segmentation ONNX Model
# Sources:
# - PINTO Model Zoo: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/109_Selfie_Segmentation
# - Hugging Face: https://huggingface.co/onnx-community/mediapipe_selfie_segmentation

set -e

echo "=== Downloading MediaPipe Selfie Segmentation Model ==="
echo ""

# Try multiple sources
MODEL_DIR="$(dirname "$0")"
cd "$MODEL_DIR"

# Option 1: Try direct download from GitHub raw content (PINTO Model Zoo)
echo "Attempting to download from PINTO Model Zoo..."
PINTO_URL="https://github.com/PINTO0309/PINTO_model_zoo/raw/main/109_Selfie_Segmentation/saved_model_256x256/model_float32.onnx"

if curl -L -f -o segmentation.onnx "$PINTO_URL" 2>/dev/null; then
    echo "✓ Successfully downloaded from PINTO Model Zoo"
    ls -lh segmentation.onnx
    exit 0
fi

echo "✗ PINTO URL not accessible, trying alternative..."

# Option 2: Try alternative ONNX model (lightweight version)
echo "Attempting alternative download..."
ALT_URL="https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float32/latest/selfie_segmenter.tflite"

if curl -L -f -o selfie_segmenter.tflite "$ALT_URL" 2>/dev/null; then
    echo "✓ Downloaded TFLite model"
    echo "⚠️  You'll need to convert TFLite to ONNX"
    echo "Run: pip install tf2onnx tensorflow"
    echo "Then convert the model using the conversion script"
    exit 0
fi

echo ""
echo "=== Automatic download failed ==="
echo ""
echo "Please download manually from one of these sources:"
echo ""
echo "1. PINTO Model Zoo (Recommended):"
echo "   https://github.com/PINTO0309/PINTO_model_zoo/tree/main/109_Selfie_Segmentation"
echo "   - Navigate to saved_model_256x256/"
echo "   - Download model_float32.onnx"
echo "   - Rename to segmentation.onnx"
echo ""
echo "2. Hugging Face:"
echo "   https://huggingface.co/onnx-community/mediapipe_selfie_segmentation"
echo "   - Click 'Files and versions'"
echo "   - Download the ONNX model file"
echo ""
echo "3. Try wget or git clone:"
echo "   git clone https://github.com/PINTO0309/PINTO_model_zoo.git --depth 1"
echo "   cp PINTO_model_zoo/109_Selfie_Segmentation/saved_model_256x256/*.onnx segmentation.onnx"
echo ""

exit 1
