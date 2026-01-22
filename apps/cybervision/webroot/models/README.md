# Segmentation Model Setup

The segmentation effect requires an ONNX model file to function.

## Quick Setup

1. Download a portrait segmentation ONNX model (see `DOWNLOAD_MODEL.md` for options)
2. Place it in this directory as `segmentation.onnx`
3. Restart the player and select "Segmentation" effect

## Model Requirements

- Input shape: `[1, 256, 256, 3]` (NHWC format)
- Input type: float32, range [0, 1]
- Output shape: `[1, 1, 256, 256]` or `[1, 256, 256]`
- Output type: float32, range [0, 1]

## Testing Without Model

The segmentation effect will show passthrough video if no model is found. Download and place the model file to enable actual segmentation.
