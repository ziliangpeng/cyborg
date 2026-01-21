// Glitch compute shader
// Modes: 0=Slices, 1=Blocks, 2=Scanlines

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct GlitchParams {
  mode: f32,
  intensity: f32,
  blockSize: f32,
  colorShift: f32,
  noiseAmount: f32,
  scanlineStrength: f32,
  time: f32,
  _pad0: f32,
  width: f32,
  height: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(2) var<uniform> params: GlitchParams;

fn hash(p: vec2f) -> f32 {
  let p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  let p3_dot = dot(p3, vec3f(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
  let p3_result = p3 + vec3f(p3_dot, p3_dot, p3_dot);
  return fract((p3_result.x + p3_result.y) * p3_result.z);
}

fn samplePixel(pixel: vec2f) -> vec3f {
  if (pixel.x < 0.0 || pixel.x >= params.width || pixel.y < 0.0 || pixel.y >= params.height) {
    return vec3f(0.0);
  }
  let ip = vec2i(i32(pixel.x), i32(pixel.y));
  return textureLoad(inputTex, ip, 0).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let pos = vec2f(f32(id.x), f32(id.y));
  let mode = i32(params.mode);
  let blockSize = max(params.blockSize, 1.0);

  var shiftedPos = pos;

  if (mode == 0) {
    let band = floor(pos.y / blockSize);
    let offset = (hash(vec2f(band, params.time)) - 0.5) * 2.0 * params.intensity;
    shiftedPos = pos + vec2f(offset, 0.0);
  } else if (mode == 1) {
    let block = floor(pos / blockSize);
    let base = vec2f(block.x, block.y);
    let offsetX = (hash(base + vec2f(params.time, 13.7)) - 0.5) * 2.0 * params.intensity;
    let offsetY = (hash(base + vec2f(17.3, params.time)) - 0.5) * 2.0 * params.intensity;
    shiftedPos = pos + vec2f(offsetX, offsetY);
  } else {
    let offset = (hash(vec2f(pos.y, params.time)) - 0.5) * 2.0 * (params.intensity * 0.5);
    shiftedPos = pos + vec2f(offset, 0.0);
  }

  let shift = params.colorShift;
  let r = samplePixel(shiftedPos + vec2f(shift, 0.0)).r;
  let g = samplePixel(shiftedPos).g;
  let b = samplePixel(shiftedPos - vec2f(shift, 0.0)).b;

  var color = vec3f(r, g, b);

  if (params.scanlineStrength > 0.0) {
    let line = 0.5 + 0.5 * sin((pos.y + params.time * 60.0) * 3.14159);
    color *= 1.0 - params.scanlineStrength * line;
  }

  if (params.noiseAmount > 0.0) {
    let noise = (hash(pos + vec2f(params.time * 10.0, params.time * 37.0)) - 0.5) * 2.0;
    color += noise * params.noiseAmount;
  }

  color = clamp(color, vec3f(0.0), vec3f(1.0));
  textureStore(outputTex, vec2i(id.xy), vec4f(color, 1.0));
}
