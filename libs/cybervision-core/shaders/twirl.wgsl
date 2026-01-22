// Twirl warp compute shader

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

struct TwirlParams {
  centerX: f32,
  centerY: f32,
  radius: f32,
  strength: f32,
}

@group(0) @binding(2) var<uniform> params: TwirlParams;

fn bilinearSample(tex: texture_2d<f32>, uv: vec2f, dims: vec2i) -> vec4f {
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

  let coord = vec2i(i32(id.x), i32(id.y));
  let pos = vec2f(f32(coord.x), f32(coord.y));
  let dimsF = vec2f(f32(dims.x), f32(dims.y));
  let uv = pos / dimsF;
  let center = vec2f(params.centerX, params.centerY);

  let offset = uv - center;
  let dist = length(offset);

  var sourceUv = uv;
  if (params.radius > 0.0 && dist < params.radius) {
    let percent = (params.radius - dist) / params.radius;
    let angle = params.strength * percent;
    let sinA = sin(angle);
    let cosA = cos(angle);
    let rotated = vec2f(
      offset.x * cosA - offset.y * sinA,
      offset.x * sinA + offset.y * cosA
    );
    sourceUv = center + rotated;
  }

  let sourcePos = sourceUv * dimsF;
  let color = bilinearSample(inputTex, sourcePos, vec2i(i32(dims.x), i32(dims.y)));
  textureStore(outputTex, coord, color);
}
