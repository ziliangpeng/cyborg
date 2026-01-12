// Chromatic aberration compute shader
// Separates and offsets RGB channels for glitch/lens distortion effect

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct ChromaticParams {
  intensity: f32,   // Pixel offset amount
  mode: f32,        // 0=Radial, 1=Horizontal, 2=Vertical
  centerX: f32,     // 0-1 normalized
  centerY: f32,     // 0-1 normalized
  width: f32,
  height: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(2) var<uniform> params: ChromaticParams;

// Sample with bounds checking
fn sampleChannel(uv: vec2f) -> f32 {
  let dims = vec2f(params.width, params.height);
  let pixel = uv * dims;

  if (pixel.x < 0.0 || pixel.x >= params.width || pixel.y < 0.0 || pixel.y >= params.height) {
    return 0.0;
  }

  let iPixel = vec2i(i32(pixel.x), i32(pixel.y));
  return textureLoad(inputTex, iPixel, 0).r;
}

fn sampleChannelVec(uv: vec2f, channel: i32) -> f32 {
  let dims = vec2f(params.width, params.height);
  let pixel = uv * dims;

  if (pixel.x < 0.0 || pixel.x >= params.width || pixel.y < 0.0 || pixel.y >= params.height) {
    return 0.0;
  }

  let iPixel = vec2i(i32(pixel.x), i32(pixel.y));
  let color = textureLoad(inputTex, iPixel, 0);

  if (channel == 0) {
    return color.r;
  } else if (channel == 1) {
    return color.g;
  } else {
    return color.b;
  }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);

  // Check bounds
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  // Calculate UV coordinates (0-1)
  let uv = vec2f(f32(id.x) / params.width, f32(id.y) / params.height);

  // Calculate offset direction based on mode
  var offsetDir: vec2f;
  let mode = i32(params.mode);

  if (mode == 0) {
    // Radial mode - direction from center to pixel
    let center = vec2f(params.centerX, params.centerY);
    let dir = uv - center;
    let dist = length(dir);
    if (dist > 0.0) {
      offsetDir = normalize(dir);
    } else {
      offsetDir = vec2f(0.0);
    }
  } else if (mode == 1) {
    // Horizontal mode
    offsetDir = vec2f(1.0, 0.0);
  } else {
    // Vertical mode
    offsetDir = vec2f(0.0, 1.0);
  }

  // Calculate offset in UV space
  let pixelOffset = params.intensity;
  let uvOffset = offsetDir * pixelOffset / vec2f(params.width, params.height);

  // Sample each channel with offset
  let rUV = uv + uvOffset;
  let gUV = uv;
  let bUV = uv - uvOffset;

  let r = sampleChannelVec(rUV, 0);
  let g = sampleChannelVec(gUV, 1);
  let b = sampleChannelVec(bUV, 2);

  textureStore(outputTex, vec2i(id.xy), vec4f(r, g, b, 1.0));
}
