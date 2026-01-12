# Dominant Color Sampling Implementation

## Overview

The mosaic effect supports a "Dominant (Cartoon)" mode that samples the most common color in each mosaic block, creating a posterized/cartoon-like appearance. This document explains the implementation challenges and why different approaches are used for WebGPU vs WebGL.

## The Challenge

Implementing true dominant color sampling requires:

1. **Sampling all pixels in a block** - For a 32x32 block, that's 1,024 pixels
2. **Quantizing colors** - Reducing color space to make histogram tractable (e.g., 8 levels per RGB channel = 512 bins)
3. **Building a histogram** - Counting occurrences of each quantized color
4. **Finding the maximum** - Determining which color appears most frequently
5. **Writing the result** - Applying the dominant color to all pixels in the block

The core problem: **This computation must happen once per mosaic block, but standard fragment/compute shaders operate per-pixel.**

## Why It's Hard on WebGL

WebGL uses **fragment shaders** that run independently for each pixel:

### Fragment Shader Limitations

```glsl
// Fragment shader - runs ONCE PER PIXEL
void main() {
  // Each pixel needs to determine its block
  vec2 blockIndex = floor(pixelPos / blockSize);

  // Problem: To find dominant color, THIS PIXEL must:
  // 1. Sample all 1,024 pixels in its block
  // 2. Build a 512-entry histogram (2KB+ per pixel!)
  // 3. Find the max - all by itself!

  // This is EXTREMELY expensive and redundant:
  // - Every pixel in the same block repeats the same work
  // - For a 32x32 block = 1,024 pixels doing identical work
  // - Total: 1,024 pixels × 1,024 samples = 1,048,576 texture reads per block!
}
```

### Specific Limitations

1. **No shared memory** - Fragment shaders can't share data with neighboring fragments
2. **No synchronization** - Can't coordinate work between fragments
3. **Limited memory** - Can't allocate large arrays (512-entry histogram per pixel = excessive register pressure)
4. **Massive redundancy** - Every pixel in a block independently does the same histogram computation

### Result

For WebGL, we use a **simplified fallback**: `centerSample()` - just taking the center pixel of each block. This is fast but doesn't give true dominant color behavior.

## Why It's Feasible on WebGPU

WebGPU compute shaders have **workgroup shared memory** that allows threads to cooperate:

### Workgroup Architecture

```wgsl
// Shared across all 64 threads in the workgroup
var<workgroup> histogram: array<atomic<u32>, 512>;
var<workgroup> maxCount: atomic<u32>;
var<workgroup> dominantColorIndex: atomic<u32>;

@compute @workgroup_size(8, 8)  // 64 threads per workgroup
fn mainDominant(...) {
  // Key insight: Dispatch ONE WORKGROUP PER MOSAIC BLOCK
  // All 64 threads cooperate on the SAME block

  // Phase 1: Clear shared histogram (threads cooperate)
  if (localIdx < 512) {
    atomicStore(&histogram[localIdx], 0u);
  }
  workgroupBarrier();  // Wait for all threads

  // Phase 2: Each thread samples a portion of the block
  // 64 threads × 16 samples each = 1,024 total samples
  for (var s = 0; s < samplesPerThread; s++) {
    let color = samplePixel(...);
    let key = colorToKey(quantizeColor(color));
    atomicAdd(&histogram[key], 1u);  // Thread-safe increment
  }
  workgroupBarrier();  // Wait for all sampling to complete

  // Phase 3: Find maximum count (parallel reduction)
  if (localIdx < 512) {
    let count = atomicLoad(&histogram[localIdx]);
    atomicMax(&maxCount, count);
  }
  workgroupBarrier();

  // Phase 4: Find bin with max count
  // (parallel search across threads)
  workgroupBarrier();

  // Phase 5: All threads write the SAME dominant color
  let dominantColor = keyToColor(dominantIdx);
  textureStore(outputPos, vec4f(dominantColor, 1.0));
}
```

### Key Advantages

1. **Shared memory** - `var<workgroup>` lets 64 threads share a single 512-entry histogram
2. **Atomic operations** - `atomicAdd`, `atomicMax` for thread-safe histogram updates
3. **Synchronization** - `workgroupBarrier()` ensures phases complete in order
4. **Cooperative work** - 64 threads divide the sampling workload
5. **Per-block dispatch** - Compute shader dispatched once per mosaic block, not per pixel

### Dispatch Strategy

```javascript
// WebGPU: Per-block dispatch for dominant mode
const blocksX = Math.ceil(videoWidth / blockSize);
const blocksY = Math.ceil(videoHeight / blockSize);
computePass.dispatchWorkgroups(blocksX, blocksY);

// vs. other modes: Per-pixel dispatch
const workgroupsX = Math.ceil(videoWidth / 8);
const workgroupsY = Math.ceil(videoHeight / 8);
```

This means for a 640×480 video with 32px blocks:
- Dominant mode: 20×15 = **300 workgroups**
- Other modes: 80×60 = **4,800 workgroups**

But dominant mode does more work per workgroup (histogram computation), so overall performance is comparable.

## Performance Characteristics

### WebGL (centerSample fallback)
- **Cost**: O(1) per pixel - just one texture read at block center
- **Quality**: Not true dominant color, but very fast

### WebGPU (true histogram)
- **Cost**: O(blockSize²) per block, divided across 64 threads
- **Quality**: Accurate dominant color via full histogram
- **Memory**: 512 × 4 bytes = 2KB shared memory per workgroup (acceptable)

For a 32×32 block:
- 64 threads each sample ~16 pixels = 1,024 total samples
- All threads atomically update shared histogram
- Typical time: ~0.5ms per block on modern GPU

## Implementation Details

### Color Quantization

We quantize RGB to 8 levels per channel (512 total bins):

```wgsl
fn quantizeColor(color: vec3f) -> vec3i {
  return vec3i(
    i32(floor(color.r * 8.0)),
    i32(floor(color.g * 8.0)),
    i32(floor(color.b * 8.0))
  );
}

fn colorToKey(qcolor: vec3i) -> i32 {
  return qcolor.r + qcolor.g * 8 + qcolor.b * 64;
}
```

This balances:
- **Fewer bins** (e.g., 4³ = 64): Faster but loses color fidelity
- **More bins** (e.g., 16³ = 4096): Better colors but histogram too large

### Handling Variable Block Sizes

The shader handles block sizes from 4px to 64px with a fixed 8×8 workgroup:

| Block Size | Pixels/Block | Strategy |
|------------|--------------|----------|
| 4×4 = 16   | 16           | 16 threads active, 48 idle (still works) |
| 8×8 = 64   | 64           | Perfect 1:1 mapping |
| 16×16 = 256| 256          | Each thread samples 4 pixels |
| 32×32 = 1024| 1024        | Each thread samples 16 pixels |
| 64×64 = 4096| 4096        | Each thread samples 64 pixels |

## Future Optimizations

Potential improvements for WebGPU implementation:

1. **Adaptive quantization** - Use fewer bins for small blocks, more for large
2. **Subsampling** - Sample every Nth pixel instead of all pixels for large blocks
3. **Parallel reduction** - Optimize max-finding with binary tree reduction
4. **Local histogram merge** - Each thread maintains local histogram, then merge

## References

- [WebGPU Compute Shader Specification](https://www.w3.org/TR/webgpu/#compute-shaders)
- [WGSL Atomic Operations](https://www.w3.org/TR/WGSL/#atomic-builtin-functions)
- [Workgroup Memory and Barriers](https://www.w3.org/TR/WGSL/#workgroup-barrier)
