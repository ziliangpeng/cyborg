// Ordered dither compute shader (4x4 Bayer matrix)

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct DitherParams {
  scale: f32,
  levels: f32,
  _pad0: f32,
  _pad1: f32,
}

@group(0) @binding(2) var<uniform> params: DitherParams;

const bayer4x4: array<f32, 16> = array<f32, 16>(
  0.0, 8.0, 2.0, 10.0,
  12.0, 4.0, 14.0, 6.0,
  3.0, 11.0, 1.0, 9.0,
  15.0, 7.0, 13.0, 5.0
);

fn luminance(color: vec3f) -> f32 {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

fn bayerThreshold(ix: i32, iy: i32) -> f32 {
  let idx = u32(ix + iy * 4);
  return (bayer4x4[idx] + 0.5) / 16.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let pos = vec2i(i32(id.x), i32(id.y));
  let color = textureLoad(inputTex, pos, 0).rgb;
  let lum = luminance(color);

  let scale = max(params.scale, 1.0);
  let levels = max(params.levels, 2.0);
  let maxLevel = levels - 1.0;

  let cell = floor(vec2f(f32(pos.x), f32(pos.y)) / scale);
  let ix = i32(cell.x) & 3;
  let iy = i32(cell.y) & 3;
  let threshold = bayerThreshold(ix, iy);

  var base = floor(lum * maxLevel);
  let frac = lum * maxLevel - base;
  if (frac > threshold) {
    base = min(base + 1.0, maxLevel);
  }

  let quant = base / maxLevel;
  textureStore(outputTex, pos, vec4f(vec3f(quant), 1.0));
}
