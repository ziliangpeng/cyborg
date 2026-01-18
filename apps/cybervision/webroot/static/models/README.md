# ML Models

This directory contains ONNX models for CyberVision ML effects.

## Portrait Segmentation Model

For portrait segmentation, you need a segmentation model that outputs person/background masks.

### Recommended Models:

1. **MediaPipe Selfie Segmentation** (256x256)
   - Download: https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float32/latest/selfie_segmenter.tflite
   - Convert to ONNX using: https://github.com/PINTO0309/tflite2tensorflow

2. **SINet Portrait Segmentation** (256x256)
   - Search for "SINet ONNX portrait segmentation" models on Hugging Face

3. **Export your own from PyTorch**:
   ```python
   import torch
   import torch.onnx

   # Assuming you have a segmentation model
   model = YourSegmentationModel()
   model.eval()

   dummy_input = torch.randn(1, 3, 256, 256)
   torch.onnx.export(
       model,
       dummy_input,
       "segmentation.onnx",
       input_names=['input'],
       output_names=['output'],
       dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
   )
   ```

### Expected Input/Output:

- **Input**: `[1, 256, 256, 3]` - RGB image normalized to [0, 1] (NHWC format)
- **Output**: `[1, 1, 256, 256]` - Single-channel mask [0, 1] or logits

### Using the Model:

Place your `segmentation.onnx` file in this directory. The app will load it from:
```
static/models/segmentation.onnx
```

### Model Requirements:

- Size: < 10MB preferred
- Format: ONNX (opset 11+)
- Input size: 256x256 (configurable in ml-inference.js)
- Output: Single channel mask
