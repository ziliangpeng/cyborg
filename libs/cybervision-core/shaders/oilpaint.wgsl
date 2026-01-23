// Oil paint effect compute shader

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct OilPaintParams {
  radius: f32,
  levels: f32,
  _pad0: f32,
  _pad1: f32,
}

@group(0) @binding(2) var<uniform> params: OilPaintParams;

const MAX_RADIUS: i32 = 6;
const MAX_LEVELS: i32 = 8;

fn luminance(color: vec3f) -> f32 {
  return dot(color, vec3f(0.299, 0.587, 0.114));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let pos = vec2i(i32(id.x), i32(id.y));
  let radius = clamp(i32(params.radius), 1, MAX_RADIUS);
  let levels = clamp(i32(params.levels), 1, MAX_LEVELS);
  let maxX = i32(dims.x) - 1;
  let maxY = i32(dims.y) - 1;

  var counts: array<i32, MAX_LEVELS>;
  var sums: array<vec3f, MAX_LEVELS>;
  for (var i = 0; i < MAX_LEVELS; i++) {
    counts[i] = 0;
    sums[i] = vec3f(0.0);
  }

  for (var dy = -MAX_RADIUS; dy <= MAX_RADIUS; dy++) {
    for (var dx = -MAX_RADIUS; dx <= MAX_RADIUS; dx++) {
      if (abs(dx) > radius || abs(dy) > radius) {
        continue;
      }

      let sampleX = clamp(pos.x + dx, 0, maxX);
      let sampleY = clamp(pos.y + dy, 0, maxY);
      let color = textureLoad(inputTex, vec2i(sampleX, sampleY), 0).rgb;
      let lum = luminance(color);
      let bucket = clamp(i32(floor(lum * f32(levels - 1) + 0.5)), 0, levels - 1);

      counts[bucket] = counts[bucket] + 1;
      sums[bucket] = sums[bucket] + color;
    }
  }

  var maxCount = 0;
  var maxIndex = 0;
  for (var i = 0; i < MAX_LEVELS; i++) {
    if (i >= levels) {
      break;
    }
    if (counts[i] > maxCount) {
      maxCount = counts[i];
      maxIndex = i;
    }
  }

  let denom = max(1, maxCount);
  let outColor = sums[maxIndex] / f32(denom);
  textureStore(outputTex, pos, vec4f(outColor, 1.0));
}
