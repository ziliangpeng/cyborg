# Portrait Segmentation Implementation Summary

Implementation of ML-based portrait segmentation effect in CyberVision, completed in 6 phases.

## Overview

Added real-time portrait segmentation as the first ML-powered effect in CyberVision, with three compositing modes:

1. **Blur Background** - Gaussian blur on background, person stays sharp
2. **Replace Background** - Swap background with uploaded image
3. **Black Out Person** - Person becomes silhouette, background unchanged

## Technical Architecture

### Runtime
- **ML Framework**: ONNX Runtime Web v1.20.1
- **Execution Providers**: WebGPU (preferred) → WASM (fallback)
- **Model Input**: 256x256 RGB normalized to [-1, 1]
- **Model Output**: 256x256 single-channel mask
- **Expected Performance**: 30+ fps on M4 Pro (with frame skipping)

### Model Requirements

The segmentation model should be placed at:
```
apps/cybervision/webroot/static/models/segmentation.onnx
```

See `apps/cybervision/webroot/static/models/README.md` for model recommendations and requirements.

**Expected Model Specs:**
- Input: `[1, 256, 256, 3]` - RGB image normalized to [0, 1] (NHWC format)
- Output: `[1, 1, 256, 256]` - Single-channel mask [0, 1] or logits
- Size: < 10MB preferred
- Format: ONNX (opset 11+)

## Implementation Details

### Phase 1: ONNX Runtime Setup
**Files Modified:**
- `apps/cybervision/package.json` - Added `onnxruntime-web` dependency
- `apps/cybervision/webroot/static/ml-inference.js` - Created ML inference wrapper

**Key Features:**
- WebGPU detection with WASM fallback
- Progress tracking during model download
- Async model loading with callbacks

### Phase 2: Segmentation Model Integration
**Files Modified:**
- `apps/cybervision/webroot/static/ml-inference.js` - Added `PortraitSegmentation` class
- `apps/cybervision/webroot/static/models/README.md` - Created model documentation

**Key Features:**
- Preprocessing: Resize to 256x256, normalize to [-1, 1], convert to CHW format
- Postprocessing: Sigmoid activation, thresholding, binary mask generation
- Upsampling: Bilinear interpolation to target resolution
- Debug visualization support

### Phase 3: WebGPU Compute Shader
**Files Created:**
- `apps/cybervision/webroot/static/shaders/segmentation.wgsl`

**Shader Features:**
- Three compositing modes (blur, replace, blackout)
- Configurable Gaussian blur radius
- Edge feathering for smooth transitions
- Mask threshold control
- Box blur implementation for performance

**Shader Bindings:**
- `@binding(0)` - Input texture (video frame)
- `@binding(1)` - Mask texture (from ML)
- `@binding(2)` - Background texture (for replace mode)
- `@binding(3)` - Output texture (storage)
- `@binding(4)` - Uniform buffer (parameters)

### Phase 4: Renderer Integration
**Files Modified:**
- `apps/cybervision/webroot/static/webgpu-renderer.js`

**Changes:**
- Added segmentation pipeline, shader module, and bind groups
- Created mask texture (256x256) and background texture (video size)
- Implemented `updateSegmentationMask()` - Updates mask from ML inference
- Implemented `updateBackgroundImage()` - Loads background image for replace mode
- Implemented `renderSegmentation()` - Main render method with mode switching

**Performance Optimizations:**
- Mask texture is 256x256 (upsampled in shader)
- Background texture resizing with aspect-ratio-preserving crop
- Efficient GPU texture updates

### Phase 5: UI Integration
**Files Modified:**
- `apps/cybervision/webroot/index.html`
- `apps/cybervision/webroot/static/app.js`

**UI Components Added:**
- Effect button "SE" (Segmentation) in Artistic tab
- Mode dropdown (Blur/Replace/Blackout)
- Blur radius slider (0-30px)
- Background image upload with preview
- Loading indicator with progress
- Info text about model requirements

**App.js Integration:**
- Added DOM element references for all controls
- Added state variables for ML model, mask, background image
- Implemented event handlers for mode changes and uploads
- Added `loadSegmentationModel()` - Lazy-loads model on first use
- Added `updateSegmentationControlsVisibility()` - Shows/hides relevant controls
- Added `renderSegmentation()` - Async render with frame skipping

### Phase 6: Performance Optimizations
**Optimizations Implemented:**
- **Frame Skipping**: Run inference every 2-3 frames, reuse mask
- **Lazy Loading**: Model loads only when effect is first selected
- **Progressive Loading**: Show progress during model download
- **Mask Caching**: Reuse previous mask while new inference runs
- **Texture Size**: 256x256 mask (not full video resolution)
- **Async Inference**: Non-blocking ML inference
- **WASM Fallback**: Functional without WebGPU (slower)

## Usage

### For Users

1. **Select Effect**: Click "SE" button in the Artistic tab
2. **Wait for Model**: First use downloads ~5MB model (shows progress)
3. **Choose Mode**:
   - Blur Background: Adjust blur radius slider
   - Replace Background: Upload image file
   - Blackout Person: No additional controls
4. **Performance**: Should maintain 25-30+ fps on modern hardware

### For Developers

**Adding a Model:**
```bash
# Place your ONNX model at:
apps/cybervision/webroot/static/models/segmentation.onnx
```

**Testing Without Model:**
Effect will fall back to passthrough (original video) if model is missing.

**Adjusting Frame Skip:**
```javascript
// In app.js constructor:
this.segmentationFrameSkip = 2;  // Run every 2nd frame (default)
```

**Modifying Blur Radius Range:**
```html
<!-- In index.html -->
<input
  id="segmentationBlurRadius"
  type="range"
  min="0"
  max="30"  <!-- Adjust max here -->
  value="10"
  step="1"
/>
```

## Dependencies

### NPM Package
- `onnxruntime-web@1.20.1` - Added to package.json

### CDN Scripts
- ONNX Runtime Web loaded via CDN in index.html

## File Changes Summary

| File | Type | Description |
|------|------|-------------|
| `package.json` | Modified | Added onnxruntime-web dependency |
| `ml-inference.js` | Created | ML inference wrapper and segmentation class |
| `models/README.md` | Created | Model documentation |
| `models/segmentation.onnx` | Required | ONNX model (not included, see README) |
| `shaders/segmentation.wgsl` | Created | WebGPU compute shader |
| `webgpu-renderer.js` | Modified | Added pipeline, textures, render methods |
| `index.html` | Modified | Added UI controls and ONNX script tag |
| `app.js` | Modified | Added state, event handlers, render logic |

## Known Limitations

1. **WebGPU Only**: Segmentation effect requires WebGPU (falls back to passthrough on WebGL)
2. **Model Required**: Effect won't work without model file (shows loading error)
3. **Initial Load**: First use downloads 5MB model (subsequent uses are instant)
4. **Performance**: Frame skipping needed for real-time performance on slower devices
5. **Mask Quality**: 256x256 mask resolution may show jagged edges (mitigated by feathering)

## Future Enhancements

Potential improvements (not implemented):

- IndexedDB caching for model file
- Multiple segmentation models (hair, clothes, etc.)
- Fine-tuned edge refinement
- Background blur bokeh effects
- WebGL fallback implementation
- Adaptive frame skipping based on FPS
- WebWorker for inference (off main thread)

## Testing Checklist

- [ ] Model loads successfully (check console for "Model loaded")
- [ ] Progress indicator shows during download
- [ ] All three modes work (blur, replace, blackout)
- [ ] Blur radius slider changes effect
- [ ] Background image upload works
- [ ] Preview shows uploaded image
- [ ] FPS stays above 25 fps
- [ ] No memory leaks during extended use
- [ ] WebGPU fallback to WASM works
- [ ] WebGL fallback shows passthrough

## Architecture Diagram

```
User Video Input
       ↓
ML Inference (ONNX Runtime)
       ↓
Segmentation Mask (256x256)
       ↓
WebGPU Renderer
       ↓
Compositing Shader
   ↓    ↓    ↓
  Blur Replace Blackout
       ↓
Canvas Output
```

## Performance Metrics

Expected performance on M4 Pro MacBook:

- **Model Load Time**: 2-5 seconds (first time only)
- **Inference Time**: 15-30ms per frame (with WebGPU)
- **Render Time**: 5-10ms per frame
- **Total FPS**: 30+ fps (with frame skip = 2)
- **Memory Usage**: ~150MB for model + textures

## Credits

Implementation based on:
- ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/
- WebGPU Compute Shaders: https://webgpu.github.io/
- MediaPipe Segmentation: https://developers.google.com/mediapipe/solutions/vision/image_segmenter
