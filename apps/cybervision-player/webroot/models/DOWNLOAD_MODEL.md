# Download Segmentation Model

The portrait segmentation effect requires an ONNX model file. Here are quick ways to get started:

## Option 1: Download Pre-converted Model (Recommended)

Download a pre-converted ONNX segmentation model:

### MediaPipe Selfie Segmentation (Recommended)

```bash
# From the cyborg root directory
cd apps/cybervision/webroot/static/models/

# Download from a trusted source (example - replace with actual URL)
# Note: You'll need to find and download a 256x256 selfie segmentation model
# Search for: "mediapipe selfie segmentation onnx" or "portrait segmentation onnx 256x256"

# Option: Use onnx model zoo or huggingface
# Example URL (verify before using):
# wget https://huggingface.co/.../segmentation.onnx
```

## Option 2: Convert TensorFlow Lite Model

If you have MediaPipe's `.tflite` model:

```bash
pip install tf2onnx tensorflow

# Convert TFLite to ONNX
python -c "
import tensorflow as tf
import tf2onnx

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='selfie_segmenter.tflite')
interpreter.allocate_tensors()

# Convert to ONNX
# ... (full conversion script)
"
```

## Option 3: Export from PyTorch

```python
import torch
import torch.onnx

# Load your segmentation model
model = YourSegmentationModel()
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 256, 256)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "segmentation.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=11
)
```

## Model Requirements

Your model MUST have:
- **Input shape**: `[1, 256, 256, 3]` (NHWC format)
- **Input type**: float32
- **Input range**: [0, 1] (normalized RGB)
- **Output shape**: `[1, 1, 256, 256]` or `[1, 256, 256]`
- **Output type**: float32
- **Output range**: [0, 1] (probabilities) or logits (will apply sigmoid)

## Quick Test Models

### Option A: Use U-Net Based Models
Search Hugging Face for pre-trained portrait segmentation models:
- https://huggingface.co/models?search=portrait+segmentation+onnx

### Option B: Use SINet Models
SINet models are designed for portrait segmentation:
- Search for "SINet portrait segmentation ONNX"

### Option C: Export from Existing Projects
Many open-source projects provide segmentation models:
- MODNet: https://github.com/ZHKKKe/MODNet
- BackgroundMattingV2: https://github.com/PeterL1n/BackgroundMattingV2
- DeepLabV3: Available in torchvision

## Verify Your Model

Once downloaded, test it:

```bash
cd apps/cybervision/webroot/static/models/
ls -lh segmentation.onnx

# Should show a file around 3-10MB
```

## Troubleshooting

**Model too large (>10MB)?**
- Look for quantized versions (INT8, FP16)
- Use model compression tools

**Wrong input/output shapes?**
- Modify `ml-inference.js` to match your model's dimensions
- Update `modelWidth` and `modelHeight` in `PortraitSegmentation` class

**Performance issues?**
- Reduce model complexity
- Increase frame skip: `this.segmentationFrameSkip = 3` in app.js
- Use smaller input resolution

## Example: Download from Google Drive (if you have a model there)

```bash
# Install gdown
pip install gdown

# Download (replace FILE_ID)
gdown https://drive.google.com/uc?id=FILE_ID -O segmentation.onnx
```

Once you have the model file at `segmentation.onnx`, refresh the browser and the effect will work!
