// Mosaic/Pixelate compute shader
// Supports 6 sampling modes: Center, Average, Min, Max, Dominant, Random

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct MosaicParams {
  blockSize: f32,
  mode: f32,        // 0=Center, 1=Average, 2=Min, 3=Max, 4=Dominant, 5=Random
  width: f32,
  height: f32,
  time: f32,        // For random mode animation
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
}

@group(0) @binding(2) var<uniform> params: MosaicParams;

// Workgroup shared memory for dominant color mode
var<workgroup> histogram: array<atomic<u32>, 512>;
var<workgroup> maxCount: atomic<u32>;
var<workgroup> dominantColorIndex: atomic<u32>;

// Hash function for random sampling
fn hash(p: vec2f) -> f32 {
  let p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  let p3_dot = dot(p3, vec3f(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
  let p3_result = p3 + vec3f(p3_dot, p3_dot, p3_dot);
  return fract((p3_result.x + p3_result.y) * p3_result.z);
}

// Convert RGB to grayscale for luminance comparison
fn luminance(color: vec3f) -> f32 {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

// Quantize color for dominant color mode (8 levels per channel = 512 total colors)
fn quantizeColor(color: vec3f) -> vec3i {
  let levels = 8;
  return vec3i(
    min(i32(floor(color.r * f32(levels))), levels - 1),
    min(i32(floor(color.g * f32(levels))), levels - 1),
    min(i32(floor(color.b * f32(levels))), levels - 1)
  );
}

// Convert quantized color to hash key
fn colorToKey(qcolor: vec3i) -> i32 {
  return qcolor.r + qcolor.g * 8 + qcolor.b * 64;
}

// Convert key back to color (inverse of quantization)
fn keyToColor(key: i32) -> vec3f {
  let r = key % 8;
  let g = (key / 8) % 8;
  let b = key / 64;
  return vec3f(
    (f32(r) + 0.5) / 8.0,
    (f32(g) + 0.5) / 8.0,
    (f32(b) + 0.5) / 8.0
  );
}

// Sample pixel with bounds checking
fn samplePixel(pos: vec2i) -> vec3f {
  let dims = textureDimensions(inputTex);
  if (pos.x < 0 || pos.x >= i32(dims.x) || pos.y < 0 || pos.y >= i32(dims.y)) {
    return vec3f(0.0);
  }
  return textureLoad(inputTex, pos, 0).rgb;
}

// Center mode: sample center pixel of block
fn centerSample(blockTopLeft: vec2i, blockSize: i32) -> vec3f {
  let centerOffset = blockSize / 2;
  let centerPos = blockTopLeft + vec2i(centerOffset, centerOffset);
  return samplePixel(centerPos);
}

// Average mode: average all pixels in block
fn averageSample(blockTopLeft: vec2i, blockSize: i32) -> vec3f {
  var sum = vec3f(0.0);
  var count = 0.0;

  // Limit samples to prevent GPU timeout
  let maxSamples = min(blockSize, 32);

  for (var dy = 0; dy < maxSamples; dy++) {
    for (var dx = 0; dx < maxSamples; dx++) {
      let pos = blockTopLeft + vec2i(dx, dy);
      let color = samplePixel(pos);
      sum += color;
      count += 1.0;
    }
  }

  return sum / max(count, 1.0);
}

// Min mode: darkest pixel in block
fn minSample(blockTopLeft: vec2i, blockSize: i32) -> vec3f {
  var minColor = vec3f(1.0);
  var minLuminance = 1.0;

  // Limit samples to prevent GPU timeout
  let maxSamples = min(blockSize, 32);

  for (var dy = 0; dy < maxSamples; dy++) {
    for (var dx = 0; dx < maxSamples; dx++) {
      let pos = blockTopLeft + vec2i(dx, dy);
      let color = samplePixel(pos);
      let lum = luminance(color);

      if (lum < minLuminance) {
        minLuminance = lum;
        minColor = color;
      }
    }
  }

  return minColor;
}

// Max mode: brightest pixel in block
fn maxSample(blockTopLeft: vec2i, blockSize: i32) -> vec3f {
  var maxColor = vec3f(0.0);
  var maxLuminance = 0.0;

  // Limit samples to prevent GPU timeout
  let maxSamples = min(blockSize, 32);

  for (var dy = 0; dy < maxSamples; dy++) {
    for (var dx = 0; dx < maxSamples; dx++) {
      let pos = blockTopLeft + vec2i(dx, dy);
      let color = samplePixel(pos);
      let lum = luminance(color);

      if (lum > maxLuminance) {
        maxLuminance = lum;
        maxColor = color;
      }
    }
  }

  return maxColor;
}

// Dominant mode: most common quantized color in block
fn dominantSample(blockTopLeft: vec2i, blockSize: i32) -> vec3f {
  // Simplified approach: just use center sample for now
  // Full histogram is too expensive and causes issues
  return centerSample(blockTopLeft, blockSize);
}

// Random mode: random pixel from block (animated with time)
fn randomSample(blockTopLeft: vec2i, blockSize: i32) -> vec3f {
  // Use block position and time to generate random offset
  let blockPos = vec2f(f32(blockTopLeft.x), f32(blockTopLeft.y));
  let seed = blockPos + vec2f(params.time, params.time * 0.7);

  let randX = hash(seed) * f32(blockSize);
  let randY = hash(seed + vec2f(100.0, 200.0)) * f32(blockSize);

  let randomOffset = vec2i(i32(randX), i32(randY));
  let randomPos = blockTopLeft + randomOffset;

  return samplePixel(randomPos);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  let pos = vec2i(i32(id.x), i32(id.y));

  // Check bounds
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let blockSize = i32(params.blockSize);
  let mode = i32(params.mode);

  // Determine which block this pixel belongs to
  let blockX = pos.x / blockSize;
  let blockY = pos.y / blockSize;
  let blockTopLeft = vec2i(blockX * blockSize, blockY * blockSize);

  // Sample based on mode
  var color: vec3f;

  if (mode == 0) {
    color = centerSample(blockTopLeft, blockSize);
  } else if (mode == 1) {
    color = averageSample(blockTopLeft, blockSize);
  } else if (mode == 2) {
    color = minSample(blockTopLeft, blockSize);
  } else if (mode == 3) {
    color = maxSample(blockTopLeft, blockSize);
  } else if (mode == 4) {
    color = dominantSample(blockTopLeft, blockSize);
  } else {
    color = randomSample(blockTopLeft, blockSize);
  }

  textureStore(outputTex, pos, vec4f(color, 1.0));
}

// Dominant mode entry point - uses per-block dispatch with shared histogram
@compute @workgroup_size(8, 8)
fn mainDominant(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let blockSize = i32(params.blockSize);
  let localIdx = local_id.x + local_id.y * 8u;

  // Phase 1: Initialize histogram (threads cooperate to clear it)
  if (localIdx < 512u) {
    atomicStore(&histogram[localIdx], 0u);
  }
  if (localIdx == 0u) {
    atomicStore(&maxCount, 0u);
    atomicStore(&dominantColorIndex, 999999u);  // Initialize to large value for atomicMin
  }
  workgroupBarrier();

  // Phase 2: Sample pixels and build histogram
  // Each workgroup handles one mosaic block
  let blockTopLeft = vec2i(
    i32(workgroup_id.x) * blockSize,
    i32(workgroup_id.y) * blockSize
  );

  // Distribute sampling across 64 threads
  let totalPixels = blockSize * blockSize;
  let samplesPerThread = (totalPixels + 63) / 64;

  for (var s = 0; s < samplesPerThread; s++) {
    let sampleIdx = i32(localIdx) * samplesPerThread + s;
    if (sampleIdx < totalPixels) {
      let dx = sampleIdx % blockSize;
      let dy = sampleIdx / blockSize;
      let samplePos = blockTopLeft + vec2i(dx, dy);

      // Sample and quantize color
      let color = samplePixel(samplePos);
      let qcolor = quantizeColor(color);
      let key = colorToKey(qcolor);

      // Atomically increment histogram bin (with bounds check)
      if (key >= 0 && key < 512) {
        atomicAdd(&histogram[u32(key)], 1u);
      }
    }
  }
  workgroupBarrier();

  // Phase 3: Find maximum count (each thread checks its histogram bins)
  if (localIdx < 512u) {
    let count = atomicLoad(&histogram[localIdx]);
    if (count > 0u) {
      atomicMax(&maxCount, count);
    }
  }
  workgroupBarrier();

  // Phase 4: Find first bin with max count
  if (localIdx < 512u) {
    let max = atomicLoad(&maxCount);
    let count = atomicLoad(&histogram[localIdx]);
    if (count == max && count > 0u) {
      // Use atomicMin to get the lowest index with max count (deterministic)
      atomicMin(&dominantColorIndex, localIdx);
    }
  }
  workgroupBarrier();

  // Phase 5: Write output pixels (all threads write the same dominant color)
  let dominantIdx = atomicLoad(&dominantColorIndex);
  // Fallback to center sample if no dominant color found
  var outputColor: vec3f;
  if (dominantIdx == 999999u) {
    // Fallback: use center sample
    let centerOffset = blockSize / 2;
    let centerPos = blockTopLeft + vec2i(centerOffset, centerOffset);
    outputColor = samplePixel(centerPos);
  } else {
    // Use dominant quantized color
    outputColor = keyToColor(i32(dominantIdx));
  }

  // Each thread writes multiple pixels if blockSize > 8
  let pixelsPerThread = (totalPixels + 63) / 64;
  for (var p = 0; p < pixelsPerThread; p++) {
    let pixelIdx = i32(localIdx) * pixelsPerThread + p;
    if (pixelIdx < totalPixels) {
      let dx = pixelIdx % blockSize;
      let dy = pixelIdx / blockSize;
      let outputPos = blockTopLeft + vec2i(dx, dy);

      // Bounds check before writing
      if (outputPos.x < i32(params.width) && outputPos.y < i32(params.height)) {
        textureStore(outputTex, outputPos, vec4f(outputColor, 1.0));
      }
    }
  }
}
