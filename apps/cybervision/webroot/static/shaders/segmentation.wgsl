// Portrait Segmentation Shader
// Supports three modes: blur background, replace background, blackout person

struct Params {
  mode: u32,           // 0=blur, 1=replace, 2=blackout
  blurRadius: f32,     // Blur radius (for mode 0)
  threshold: f32,      // Mask threshold
  feather: f32,        // Edge feathering amount
  width: u32,
  height: u32,
  softEdges: u32,      // 0=off, 1=on
  glow: u32,           // 0=off, 1=on
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var maskTexture: texture_2d<f32>;
@group(0) @binding(2) var backgroundTexture: texture_2d<f32>;  // For replace mode
@group(0) @binding(3) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> params: Params;

// Apply box blur to a texture at given coordinates
fn boxBlur(pos: vec2<i32>) -> vec4<f32> {
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

// Sample mask with bilinear interpolation for soft edges
fn sampleMaskSoft(pos: vec2<i32>) -> f32 {
  if (params.softEdges == 0u) {
    // Standard nearest-neighbor sampling
    let maskPos = vec2<i32>(
      vec2<f32>(pos) * 256.0 / vec2<f32>(params.width, params.height)
    );
    let maskColor = textureLoad(maskTexture, maskPos, 0);
    return maskColor.r;
  }

  // Soft edges: sample neighbors and average
  let maskPosF = vec2<f32>(
    f32(pos.x) * 256.0 / f32(params.width),
    f32(pos.y) * 256.0 / f32(params.height)
  );

  let maskPos = vec2<i32>(maskPosF);
  let frac = maskPosF - vec2<f32>(maskPos);

  // Sample 4 neighbors for bilinear interpolation
  let s00 = textureLoad(maskTexture, clamp(maskPos + vec2<i32>(0, 0), vec2<i32>(0), vec2<i32>(255)), 0).r;
  let s10 = textureLoad(maskTexture, clamp(maskPos + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(255)), 0).r;
  let s01 = textureLoad(maskTexture, clamp(maskPos + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(255)), 0).r;
  let s11 = textureLoad(maskTexture, clamp(maskPos + vec2<i32>(1, 1), vec2<i32>(0), vec2<i32>(255)), 0).r;

  // Bilinear interpolation
  let s0 = mix(s00, s10, frac.x);
  let s1 = mix(s01, s11, frac.x);
  return mix(s0, s1, frac.y);
}

// Detect edge proximity for glow effect
fn detectEdge(pos: vec2<i32>, maskValue: f32) -> f32 {
  if (params.glow == 0u) {
    return 0.0;
  }

  // Sample mask at multiple offsets to detect edges
  let maskPos = vec2<i32>(
    vec2<f32>(pos) * 256.0 / vec2<f32>(params.width, params.height)
  );

  var edgeStrength = 0.0;
  let offsets = array<vec2<i32>, 8>(
    vec2<i32>(-1, -1), vec2<i32>(0, -1), vec2<i32>(1, -1),
    vec2<i32>(-1,  0),                   vec2<i32>(1,  0),
    vec2<i32>(-1,  1), vec2<i32>(0,  1), vec2<i32>(1,  1)
  );

  for (var i = 0; i < 8; i = i + 1) {
    let samplePos = clamp(
      maskPos + offsets[i] * 2,
      vec2<i32>(0),
      vec2<i32>(255)
    );
    let neighborMask = textureLoad(maskTexture, samplePos, 0).r;
    edgeStrength = max(edgeStrength, abs(maskValue - neighborMask));
  }

  // Convert edge strength to glow intensity
  // Strong edges (sharp transitions) get more glow
  return smoothstep(0.1, 0.5, edgeStrength) * 0.3;
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

  // Load mask value with optional soft edges
  let rawMaskValue = sampleMaskSoft(pos);

  // Apply feathering (adjust amount based on softEdges setting)
  var featherAmount = params.feather;
  if (params.softEdges == 1u) {
    featherAmount = max(featherAmount, 0.15);  // Increase feathering for soft edges
  }
  let lower = params.threshold - featherAmount;
  let upper = params.threshold + featherAmount;
  let maskValue = smoothstep(lower, upper, rawMaskValue);

  // Calculate edge glow
  let glowIntensity = detectEdge(pos, rawMaskValue);

  var outputColor: vec4<f32>;

  // Mode 0: Blur background
  if (params.mode == 0u) {
    let blurredColor = boxBlur(pos);
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

  // Apply glow effect at edges
  if (glowIntensity > 0.0) {
    // Soft white/blue glow
    let glowColor = vec4<f32>(0.9, 0.95, 1.0, 1.0);
    outputColor = mix(outputColor, glowColor, glowIntensity);
  }

  // Write output
  textureStore(outputTexture, pos, outputColor);
}
