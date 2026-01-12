// Image Rotation Shader
// Rotates image by arbitrary angle using bilinear interpolation

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct RotateParams {
  angle: f32,      // Rotation angle in radians
  width: f32,
  height: f32,
  _pad: f32,
}

@group(0) @binding(2) var<uniform> params: RotateParams;

fn bilinearSample(tex: texture_2d<f32>, uv: vec2f, dims: vec2i) -> vec4f {
  // Clamp to texture bounds
  let clamped = clamp(uv, vec2f(0.0), vec2f(f32(dims.x - 1), f32(dims.y - 1)));

  let x = clamped.x;
  let y = clamped.y;

  let x0 = i32(floor(x));
  let y0 = i32(floor(y));
  let x1 = min(x0 + 1, dims.x - 1);
  let y1 = min(y0 + 1, dims.y - 1);

  let fx = fract(x);
  let fy = fract(y);

  let c00 = textureLoad(tex, vec2i(x0, y0), 0);
  let c10 = textureLoad(tex, vec2i(x1, y0), 0);
  let c01 = textureLoad(tex, vec2i(x0, y1), 0);
  let c11 = textureLoad(tex, vec2i(x1, y1), 0);

  let c0 = mix(c00, c10, fx);
  let c1 = mix(c01, c11, fx);

  return mix(c0, c1, fy);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dims = textureDimensions(outputTex);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }

  let coord = vec2i(id.xy);

  // Calculate center of image
  let center = vec2f(f32(dims.x) / 2.0, f32(dims.y) / 2.0);

  // Convert output pixel to centered coordinates
  let pos = vec2f(f32(coord.x), f32(coord.y)) - center;

  // Rotate backwards (to find source position)
  let cosA = cos(-params.angle);
  let sinA = sin(-params.angle);

  let rotatedX = pos.x * cosA - pos.y * sinA;
  let rotatedY = pos.x * sinA + pos.y * cosA;

  // Convert back to texture coordinates
  let sourcePos = vec2f(rotatedX, rotatedY) + center;

  // Sample with bilinear interpolation
  let color = bilinearSample(inputTex, sourcePos, dims);

  textureStore(outputTex, coord, color);
}
