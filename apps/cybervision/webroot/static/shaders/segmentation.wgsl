// Portrait Segmentation Shader
// Supports three modes: blur background, replace background, blackout person

struct Params {
  mode: u32,           // 0=blur, 1=replace, 2=blackout
  blurRadius: f32,     // Blur radius (for mode 0)
  threshold: f32,      // Mask threshold
  feather: f32,        // Edge feathering amount
  width: u32,
  height: u32,
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var maskTexture: texture_2d<f32>;
@group(0) @binding(2) var backgroundTexture: texture_2d<f32>;  // For replace mode
@group(0) @binding(3) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> params: Params;

// Gaussian blur weights for 5x5 kernel
const GAUSSIAN_5x5: array<f32, 25> = array<f32, 25>(
  0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
  0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
  0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
  0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
  0.003765, 0.015019, 0.023792, 0.015019, 0.003765
);

// Apply Gaussian blur to a texture at given coordinates
fn gaussianBlur(pos: vec2<i32>) -> vec4<f32> {
  var blurredColor = vec4<f32>(0.0);
  let radius = i32(params.blurRadius);

  // If radius is 0, return original pixel
  if (radius == 0) {
    return textureLoad(inputTexture, pos, 0);
  }

  // Simple box blur (faster than Gaussian for larger radii)
  var totalWeight = 0.0;

  for (var dy = -radius; dy <= radius; dy = dy + 1) {
    for (var dx = -radius; dx <= radius; dx = dx + 1) {
      let samplePos = pos + vec2<i32>(dx, dy);

      // Clamp to texture bounds
      let clampedPos = clamp(
        samplePos,
        vec2<i32>(0, 0),
        vec2<i32>(i32(params.width) - 1, i32(params.height) - 1)
      );

      let weight = 1.0;
      blurredColor = blurredColor + textureLoad(inputTexture, clampedPos, 0) * weight;
      totalWeight = totalWeight + weight;
    }
  }

  return blurredColor / totalWeight;
}

// Apply edge feathering to mask
fn featherMask(maskValue: f32) -> f32 {
  if (params.feather <= 0.0) {
    return maskValue;
  }

  // Smooth step for edge feathering
  let lower = params.threshold - params.feather;
  let upper = params.threshold + params.feather;

  return smoothstep(lower, upper, maskValue);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let pos = vec2<i32>(i32(global_id.x), i32(global_id.y));

  // Bounds check
  if (pos.x >= i32(params.width) || pos.y >= i32(params.height)) {
    return;
  }

  // Load original pixel
  let originalColor = textureLoad(inputTexture, pos, 0);

  // Load mask value (grayscale, so we just use r channel)
  // Scale video coordinates to mask coordinates (256x256)
  let maskPos = vec2<i32>(
    pos.x * 256 / i32(params.width),
    pos.y * 256 / i32(params.height)
  );
  let maskColor = textureLoad(maskTexture, maskPos, 0);
  let rawMaskValue = maskColor.r;

  // Apply feathering
  let maskValue = featherMask(rawMaskValue);

  var outputColor: vec4<f32>;

  // Mode 0: Blur background
  if (params.mode == 0u) {
    let blurredColor = gaussianBlur(pos);
    // Blend: person (mask=1) = original, background (mask=0) = blurred
    outputColor = mix(blurredColor, originalColor, maskValue);
  }
  // Mode 1: Replace background
  else if (params.mode == 1u) {
    let backgroundColor = textureLoad(backgroundTexture, pos, 0);
    // Blend: person (mask=1) = original, background (mask=0) = replacement
    outputColor = mix(backgroundColor, originalColor, maskValue);
  }
  // Mode 2: Blackout person
  else if (params.mode == 2u) {
    let blackColor = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    // Blend: person (mask=1) = black, background (mask=0) = original
    outputColor = mix(originalColor, blackColor, maskValue);
  }
  // Default: pass through
  else {
    outputColor = originalColor;
  }

  // Write output
  textureStore(outputTexture, pos, outputColor);
}
