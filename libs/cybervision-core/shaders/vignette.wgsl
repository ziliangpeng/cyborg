// Vignette + film grain compute shader

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct VignetteParams {
  vignette: f32,
  grain: f32,
  time: f32,
  _pad0: f32,
}

@group(0) @binding(2) var<uniform> params: VignetteParams;

fn rand(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let pos = vec2f(f32(id.x), f32(id.y));
  let dimsF = vec2f(f32(dims.x), f32(dims.y));
  let uv = (pos + vec2f(0.5, 0.5)) / dimsF;

  let aspect = dimsF.x / dimsF.y;
  let centered = vec2f((uv.x - 0.5) * aspect, uv.y - 0.5);
  let dist = length(centered);
  let vignette = smoothstep(0.35, 0.9, dist);
  let vignetteFactor = 1.0 - vignette * params.vignette;

  let noise = (rand(pos + params.time) - 0.5) * params.grain;

  var color = textureLoad(inputTex, vec2i(i32(id.x), i32(id.y)), 0).rgb;
  color = color * vignetteFactor + noise;
  color = clamp(color, vec3f(0.0), vec3f(1.0));

  textureStore(outputTex, vec2i(i32(id.x), i32(id.y)), vec4f(color, 1.0));
}
