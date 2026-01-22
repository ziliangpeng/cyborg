// Posterize compute shader
// Quantizes luminance into discrete levels and preserves hue

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct PosterizeParams {
  levels: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(2) var<uniform> params: PosterizeParams;

fn luminance(color: vec3f) -> f32 {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
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
  let levels = max(params.levels, 2.0);
  let quant = floor(lum * (levels - 1.0) + 0.5) / (levels - 1.0);

  var outColor = color;
  if (lum > 0.0001) {
    outColor = color * (quant / lum);
  } else {
    outColor = vec3f(0.0);
  }

  outColor = clamp(outColor, vec3f(0.0), vec3f(1.0));
  textureStore(outputTex, pos, vec4f(outColor, 1.0));
}
