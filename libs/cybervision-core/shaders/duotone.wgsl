// Duotone compute shader
// Maps luminance to a two-color gradient

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct DuotoneParams {
  shadow: vec3f,
  _pad0: f32,
  highlight: vec3f,
  _pad1: f32,
}

@group(0) @binding(2) var<uniform> params: DuotoneParams;

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
  let duo = mix(params.shadow, params.highlight, lum);

  textureStore(outputTex, pos, vec4f(duo, 1.0));
}
